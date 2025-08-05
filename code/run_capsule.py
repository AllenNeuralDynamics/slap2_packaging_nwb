import h5py
import numpy as np
import pynwb
from datetime import datetime
from pathlib import Path
from openscope_upload import harp_utils


data_folder = Path("../data")
results_folder = Path("../results")

def main():
    asset_path = data_folder / "slap2_session"
    h5_path = next(asset_path.rglob("experiment_summary.h5"))
    print("Found h5 file:", h5_path)
    nwb_path = results_folder / "output.nwb"
    session_json_path = next(asset_path.glob("session.json"))
    rig_json_path = next(asset_path.glob("rig.json"))
    harp_path = next(asset_path.rglob('.harp'))
    import json
    with open(session_json_path, "r") as f:
        session_json = json.load(f)
    with open(rig_json_path, "r") as f:
        rig_json = json.load(f)
    with h5py.File(h5_path, "r") as h5:
        nwbfile, nwb_io = create_nwb_file(nwb_path)
        harp_data = harp_utils.extract_harp(harp_path)
        add_ophys_to_nwb(nwbfile, h5, rig_json, harp_data)
        nwb_io.write(nwbfile)
        nwb_io.close()


def create_nwb_file(nwb_path):
    """
    Create and return a new NWBFile and NWBHDF5IO handle for writing.
    nwb_path: Path to the output NWB file.
    Returns: (nwbfile, nwb_io)
    """
    subject_info = {}  # TODO: extract from h5 or elsewhere
    nwbfile = pynwb.NWBFile(
        session_description='Session description placeholder',
        identifier='unique_id_placeholder',
        session_start_time=datetime.now(),
        experimenter='experimenter_placeholder',
        lab='lab_placeholder',
        institution='institution_placeholder',
        session_id='session_id_placeholder',
        subject=None  # TODO: create pynwb.Subject from subject_info
    )
    nwb_io = pynwb.NWBHDF5IO(str(nwb_path), 'w')
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


def get_pixel_mask(mask):
    """
    Given a 2D mask (full FOV), return a list of (row, col, weight) triplets for all nonzero and non-nan pixels.
    This is the NWB-compliant, space-efficient way to store ROIs for large FOVs.
    Returns: pixel_mask (list of (row, col, weight))
    """
    valid = np.logical_and(mask != 0, ~np.isnan(mask))
    rows, cols = np.where(valid)
    weights = mask[rows, cols]
    pixel_mask = [(int(r), int(c), float(w)) for r, c, w in zip(rows, cols, weights)]
    return pixel_mask


def add_image_segmentation(h5, imaging_plane, dmd_name, image_segmentation):
    """
    Add a PlaneSegmentation for a DMD to the shared ImageSegmentation object.
    For each ROI, store a pixel_mask (list of (row, col, weight)) for space-efficient NWB compliance.
    h5: Open HDF5 file handle.
    imaging_plane: ImagingPlane object for this DMD.
    dmd_name: Name of the DMD/plane (e.g., 'DMD1').
    image_segmentation: Shared ImageSegmentation object for the ophys module.
    Returns: DynamicTableRegion referencing all ROIs for this plane.
    """
    ps = image_segmentation.create_plane_segmentation(
        name=f"PlaneSegmentation_{dmd_name}",
        description=f"ROIs for {dmd_name}",
        imaging_plane=imaging_plane
    )
    fp_masks = h5[dmd_name]['sources']['spatial']['fp_masks'][()]
    n_rois = fp_masks.shape[0]
    for mask in fp_masks:
        pixel_mask = get_pixel_mask(mask)
        ps.add_roi(pixel_mask=pixel_mask)
    roi_table_region = ps.create_roi_table_region(
        region=list(range(n_rois)),
        description=f"All ROIs for {dmd_name}"
    )
    return roi_table_region


def add_fluorescence(h5, dmd_name, roi_table, ophys_mod, harp_data):
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
    print("temporal sources:", temporal_sources, temporal_sources.keys())
    f0_data = temporal_sources['F0'][()]
    dff_data = temporal_sources['dFF'][()]
    print(dmd_group, dmd_group.keys())
    trial_start_idxs = dmd_group['frame_info']['trial_start_idxs'][()]
    timestamps = harp_utils.get_concatenated_timestamps(f0_data, trial_start_idxs, harp_data)
    assert len(timestamps) == f0_data.shape[0], "Timestamps and fluorescence trace must have same length"
    fluorescence = pynwb.ophys.Fluorescence(name=f"Fluorescence_{dmd_name}")
    ophys_mod.add_data_interface(fluorescence)
    rrs_f0 = pynwb.ophys.RoiResponseSeries(
        name=f"{dmd_name}_F0",
        data=f0_data,
        rois=roi_table,
        unit='a.u.',
        timestamps=timestamps,
        description=f"F0 (raw fluorescence) traces from {dmd_name}"
    )
    fluorescence.add_roi_response_series(rrs_f0)
    rrs_dff = pynwb.ophys.RoiResponseSeries(
        name=f"{dmd_name}_dFF",
        data=dff_data,
        rois=roi_table,
        unit='a.u.',
        timestamps=timestamps,
        description=f"dFF traces from {dmd_name}"
    )
    fluorescence.add_roi_response_series(rrs_dff)


def add_mean_images(h5, dmd_name, ophys_mod):
    """
    Add registered and motion-corrected mean and activity images for a DMD to the ophys ProcessingModule as ImageSeries objects.
    mean_im: shape (channels, height, width) from HDF5.
    act_im: shape (height, width) from HDF5.
    Each channel of the mean image is stored as a single-frame ImageSeries.
    The activity image is stored as a single-frame ImageSeries.
    """
    mean_im = h5[dmd_name]['visualizations']['mean_im'][()]
    act_im = h5[dmd_name]['visualizations']['act_im'][()]
    if mean_im.ndim == 2: # should be (channels, height, width), add dim if only one channel
        mean_im = mean_im[None, ...]
    for ch in range(mean_im.shape[0]):
        mean_image_series = pynwb.image.ImageSeries(
            name=f"{dmd_name}_mean_image_channel{ch}",
            data=mean_im[ch][None, ...],  # shape (1, height, width)
            unit='a.u.',
            format='raw',
            timestamps=[0.0],
            description=f"Registered mean image for {dmd_name}, channel {ch} (motion corrected)"
        )
        ophys_mod.add_data_interface(mean_image_series)
    act_image_series = pynwb.image.ImageSeries(
        name=f"{dmd_name}_activity_image",
        data=act_im[None, ...],  # shape (1, height, width)
        unit='a.u.',
        format='raw',
        timestamps=[0.0],
        description=f"Registered activity image for {dmd_name} (motion corrected)"
    )
    ophys_mod.add_data_interface(act_image_series)


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


def add_ophys_to_nwb(nwbfile, h5, rig_json, harp_data):
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
        add_fluorescence(h5, dmd_name, roi_table, ophys_mod, harp_data)
        add_mean_images(h5, dmd_name, ophys_mod)


if __name__ == "__main__":
    main()