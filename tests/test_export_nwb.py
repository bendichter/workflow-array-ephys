import datetime

import numpy as np
from pynwb.ecephys import ElectricalSeries

from workflow_array_ephys.pipeline import ecephys_session_to_nwb, write_nwb, session_to_nwb


def test_session_to_nwb():

    nwbfile = session_to_nwb(
        {
            "subject": "subject5",
            "session_datetime": datetime.datetime(2020, 5, 12, 4, 13, 7),
        },
        lab_key={"lab": "LabA"},
        protocol_key={"protocol": "ProtA"},
        project_key={"project": "ProjA"},
    )

    assert nwbfile.session_id == "subject5_2020-05-12T04:13:07"
    assert nwbfile.session_description == "Test"
    assert nwbfile.session_start_time == datetime.datetime(
        2020, 5, 12, 4, 13, 7, tzinfo=datetime.timezone.utc
    )
    assert nwbfile.experimenter == ["User1"]

    assert nwbfile.subject.subject_id == "subject5"
    assert nwbfile.subject.sex == "M"

    assert nwbfile.institution == "Example Uni"
    assert nwbfile.lab == "The Example Lab"

    assert nwbfile.protocol == "ProtA"
    assert nwbfile.notes == "Protocol for managing data ingestion"

    assert nwbfile.experiment_description == "Example project to populate element-lab"


def test_convert_to_nwb():

    nwbfile = ecephys_session_to_nwb(
        dict(subject="subject5", session_datetime="2020-05-12 04:13:07")
    )

    for x in ("262716621", "714000838"):
        assert x in nwbfile.devices

    assert len(nwbfile.electrodes) == 1920
    for col in ("shank", "shank_row", "shank_col"):
        assert col in nwbfile.electrodes

    for es_name in ("ElectricalSeries1", "ElectricalSeries2"):
        es = nwbfile.acquisition[es_name]
        assert isinstance(es, ElectricalSeries)
        assert es.conversion == 2.34375e-06

    # make sure the ElectricalSeries objects don't share electrodes
    assert not set(nwbfile.acquisition["ElectricalSeries1"].electrodes.data) & set(
        nwbfile.acquisition["ElectricalSeries2"].electrodes.data
    )

    assert len(nwbfile.units) == 499
    for col in ("cluster_quality_label", "spike_depths"):
        assert col in nwbfile.units

    for es_name in ("ElectricalSeries1", "ElectricalSeries2"):
        es = nwbfile.processing["ecephys"].data_interfaces["LFP"][es_name]
        assert isinstance(es, ElectricalSeries)
        assert es.conversion == 4.6875e-06
        assert es.rate == 2500.0


def test_convert_to_nwb_with_dj_lfp():
    nwbfile = ecephys_session_to_nwb(
        dict(subject="subject5", session_datetime="2020-05-12 04:13:07"),
        lfp="dj",
        spikes=False,
    )

    for es_name in ("ElectricalSeries1", "ElectricalSeries2"):
        es = nwbfile.processing["ecephys"].data_interfaces["LFP"][es_name]
        assert isinstance(es, ElectricalSeries)
        assert es.conversion == 1.0
        assert isinstance(es.timestamps, np.ndarray)