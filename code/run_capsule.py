import h5py
import numpy as np
from pynwb import NWBFile, NWBHDF5IO, ProcessingModule
from pynwb.ophys import Fluorescence, RoiResponseSeries
from datetime import datetime
from pathlib import Path
import pynwb


def main():
    asset_path = r"\\allen\aind\scratch\OpenScope\Slap2\Data\801381\801381_20250605T145742"
    asset_path = Path(asset_path)
    h5_path = next(asset_path.rglob("experiment_summary.h5"))
    print("Found h5 file:", h5_path)
    nwb_path = "./results/output.nwb"
    session_json_path = next(asset_path.glob("session.json"))
    rig_json_path = next(asset_path.glob("rig.json"))
    import json
    with open(session_json_path, "r") as f:
        session_json = json.load(f)
    with open(rig_json_path, "r") as f:
        rig_json = json.load(f)
    with h5py.File(h5_path, "r") as h5:
        nwbfile, nwb_io = create_nwb_file(nwb_path)
        add_ophys_to_nwb(nwbfile, h5, rig_json)
        nwb_io.write(nwbfile)
        nwb_io.close()


def create_nwb_file(nwb_path):
    """
    Create and return a new NWBFile and NWBHDF5IO handle for writing.
    nwb_path: Path to the output NWB file.
    Returns: (nwbfile, nwb_io)
    """
    subject_info = {}  # TODO: extract from h5 or elsewhere
    nwbfile = NWBFile(
        session_description='Session description placeholder',
        identifier='unique_id_placeholder',
        session_start_time=datetime.now(),
        experimenter='experimenter_placeholder',
        lab='lab_placeholder',
        institution='institution_placeholder',
        session_id='session_id_placeholder',
        subject=None  # TODO: create pynwb.Subject from subject_info
    )
    nwb_io = NWBHDF5IO(str(nwb_path), 'w')
    return nwbfile, nwb_io


def create_imaging_plane(nwbfile, dmd_name, optical_channel, device, rig_json):
    """
    Create an ImagingPlane object for a given DMD and add it to the NWBFile.
    nwbfile: NWBFile object to add the imaging plane to.
    dmd_name: Name of the DMD/plane (e.g., 'DMD1').
    optical_channel: OpticalChannel object for this plane.
    device: Device object for this plane.
    rig_json: Dictionary with rig metadata (expects 'light_sources' and 'detectors').
    Returns: ImagingPlane object.
    """
    excitation_lambda = rig_json["light_sources"][0]["wavelength"]
    imaging_rate = rig_json["detectors"][0]["frame_rate"]
    imaging_plane = nwbfile.create_imaging_plane(
        name=f"ImagingPlane_{dmd_name}",
        optical_channel=[optical_channel],
        description=f"Imaging plane for {dmd_name}",
        device=device,
        excitation_lambda=float(excitation_lambda) if excitation_lambda is not None else 920.0,
        imaging_rate=float(imaging_rate) if imaging_rate is not None else 30.0,
        indicator="unknown",
        location="unknown",
        manifold=None
    )
    return imaging_plane


def add_image_segmentation(h5, imaging_plane, dmd_name, image_segmentation):
    """
    Add a PlaneSegmentation for a DMD to the shared ImageSegmentation object.
    h5: Open HDF5 file handle.
    imaging_plane: ImagingPlane object for this DMD.
    dmd_name: Name of the DMD/plane (e.g., 'DMD1').
    image_segmentation: Shared ImageSegmentation object for the ophys module.
    Returns: DynamicTableRegion referencing all ROIs for this plane.
    Note: Uses fp_masks as boolean ROI masks; each mask is (height, width).
    """
    ps = image_segmentation.create_plane_segmentation(
        name=f"PlaneSegmentation_{dmd_name}",
        description=f"ROIs for {dmd_name}",
        imaging_plane=imaging_plane
    )
    fp_masks = h5[dmd_name]['sources']['spatial']['fp_masks'][()]
    fp_coords = h5[dmd_name]['sources']['spatial']['fp_coords'][()]
    n_rois = fp_masks.shape[0]
    for mask in fp_masks:
        ps.add_roi(image_mask=mask)
    roi_table_region = ps.create_roi_table_region(
        region=list(range(n_rois)),
        description=f"All ROIs for {dmd_name}"
    )
    return roi_table_region


def add_fluorescence(h5, dmd_name, roi_table, ophys_mod):
    """
    Add Fluorescence and dFF traces for a DMD to the NWBFile.
    h5: Open HDF5 file handle.
    dmd_name: Name of the DMD/plane (e.g., 'DMD1').
    roi_table: DynamicTableRegion referencing ROIs for this plane.
    ophys_mod: Ophys ProcessingModule to add the Fluorescence interface to.
    Expects F0 and dFF data as (num_timepoints, num_rois).
    """
    dmd_group = h5[dmd_name]
    temporal_sources = dmd_group['sources']['temporal']
    f0_data = temporal_sources['F0'][()]
    dff_data = temporal_sources['dFF'][()]
    timestamps = np.arange(f0_data.shape[0], dtype=float)
    fluorescence = Fluorescence(name=f"Fluorescence_{dmd_name}")
    ophys_mod.add_data_interface(fluorescence)
    rrs_f0 = RoiResponseSeries(
        name=f"{dmd_name}_F0",
        data=f0_data,
        rois=roi_table,
        unit='a.u.',
        timestamps=timestamps,
        description=f"F0 (raw fluorescence) traces from {dmd_name}"
    )
    fluorescence.add_roi_response_series(rrs_f0)
    rrs_dff = RoiResponseSeries(
        name=f"{dmd_name}_dFF",
        data=dff_data,
        rois=roi_table,
        unit='a.u.',
        timestamps=timestamps,
        description=f"dFF traces from {dmd_name}"
    )
    fluorescence.add_roi_response_series(rrs_dff)


def add_mean_images(nwbfile, h5, dmd_name):
    """
    Add mean and activity images for a DMD to the NWBFile acquisitions.
    nwbfile: NWBFile object to add images to.
    h5: Open HDF5 file handle.
    dmd_name: Name of the DMD/plane (e.g., 'DMD1').
    For mean image, each channel is stored as a separate Image object.
    For activity image, only the first channel is stored.
    """
    mean_im = h5[dmd_name]['visualizations']['mean_im'][()]
    act_im = h5[dmd_name]['visualizations']['act_im'][()]
    for ch in range(mean_im.shape[0]):
        mean_image = pynwb.image.Image(name=f"{dmd_name}_mean_image_channel{ch}", data=mean_im[ch])
        nwbfile.add_acquisition(mean_image)
    act_image = pynwb.image.Image(name=f"{dmd_name}_activity_image", data=act_im[0])
    nwbfile.add_acquisition(act_image)


def create_device(nwbfile, rig_json):
    """
    Create and return a Device object using all relevant fields from rig.json.
    nwbfile: NWBFile object to add the device to.
    rig_json: Dictionary with instrument metadata.
    Returns: Device object.
    """
    name = rig_json["instrument_id"]
    manufacturer = rig_json["manufacturer"]["name"] if isinstance(rig_json["manufacturer"], dict) else rig_json["manufacturer"]
    description = f"{rig_json['instrument_type']} microscope. Temperature control: {rig_json['temperature_control']}, Humidity control: {rig_json['humidity_control']}. Calibration date: {rig_json['calibration_date']}. Notes: {rig_json['notes']}"
    device = nwbfile.create_device(
        name=name,
        description=description,
        manufacturer=manufacturer
    )
    return device


def create_optical_channel(rig_json, dmd_name):
    """
    Create an OpticalChannel object for a DMD using rig.json fields.
    nwbfile: NWBFile object (not used, for interface consistency).
    rig_json: Dictionary with instrument metadata.
    dmd_name: Name of the DMD/plane (e.g., 'DMD1').
    Returns: OpticalChannel object.
    """
    emission_lambda = 510.0  # GCaMP emission peak
    det = rig_json["detectors"][0]
    obj = rig_json["objectives"][0]
    description = f"{det['manufacturer']} {det['model']} detector, {obj['magnification']}x {obj['manufacturer']} objective, GCaMP emission"
    return pynwb.ophys.OpticalChannel(
        name=f"OpticalChannel_{dmd_name}",
        description=description,
        emission_lambda=emission_lambda
    )


def add_ophys_to_nwb(nwbfile, h5, rig_json):
    """
    Build the full ophys structure in the NWB file, iterating over DMDs (planes).
    For each DMD, create imaging plane, ROI table, add fluorescence, and add mean images.
    nwbfile: NWBFile object to populate.
    h5: Open HDF5 file handle.
    rig_json: Dictionary with instrument metadata.
    """
    device = create_device(nwbfile, rig_json)
    ophys_mod = pynwb.ProcessingModule('ophys', 'Ophys processing module')
    nwbfile.add_processing_module(ophys_mod)
    image_segmentation = pynwb.ophys.ImageSegmentation()
    ophys_mod.add_data_interface(image_segmentation)
    for dmd_name in sorted([k for k in h5.keys() if k.startswith("DMD")]):
        optical_channel = create_optical_channel(rig_json, dmd_name)
        imaging_plane = create_imaging_plane(nwbfile, dmd_name, optical_channel, device, rig_json)
        roi_table = add_image_segmentation(h5, imaging_plane, dmd_name, image_segmentation)
        add_fluorescence(h5, dmd_name, roi_table, ophys_mod)
        add_mean_images(nwbfile, h5, imaging_plane, dmd_name)


if __name__ == "__main__":
    main()