import datajoint as dj
from element_animal import subject
from element_lab import lab
from element_session import session
from element_array_ephys import probe
from element_array_ephys import ephys_no_curation as ephys

from element_animal.subject import Subject

from element_lab.lab import Source, Lab, Protocol, User, Project
from element_session.session import Session

from .paths import get_ephys_root_data_dir, get_session_directory

if 'custom' not in dj.config:
    dj.config['custom'] = {}

db_prefix = dj.config['custom'].get('database.prefix', '')


# Activate "lab", "subject", "session" schema ----------------------------------

lab.activate(db_prefix + 'lab')

Experimenter = lab.User
subject.activate(db_prefix + 'subject', linking_module=__name__)

from element_animal.export.nwb import subject_to_nwb
from element_lab.export.nwb import element_lab_to_nwb_dict

session.activate(db_prefix + 'session', linking_module=__name__)


# Declare table "SkullReference" for use in element-array-ephys ----------------

@lab.schema
class SkullReference(dj.Lookup):
    definition = """
    skull_reference   : varchar(60)
    """
    contents = zip(['Bregma', 'Lambda'])


# Activate "ephys" schema ------------------------------------------------------
from session.export.nwb import session_to_nwb

ephys.activate(db_prefix + 'ephys', 
               db_prefix + 'probe', 
               linking_module=__name__)

# from element_array_ephys.export.nwb.nwb import ecephys_session_to_nwb, write_nwb
