"""Explicit continued resonance sheet for paths leaving the original window.

The historical outgoing rule selects open real-axis cuts at Re z. That is
appropriate inside the witness's two-open-cut neighborhood. For global
continuation the cut indices and velocity signs are retained explicitly.
This is a local analytic branch in the lower half plane, not a prescription
for arbitrary loops around every square-root branch point.
"""
import math
import numpy as np
import filling_campaign as c


def bath(config,z,vertex,width):
    p=c.parameters(config['model'],width)
    sheet=config.get('pole_sheet')
    if not sheet:return c.fullk.resonance_self_energy(z,vertex,p)
    even,odd=c.fullk.auxiliary_terms(z,vertex,p)
    q=z-even;roots=c.fullk.radial_roots(q,odd,p)
    density=np.zeros((2,2),complex)
    for channel in sheet['channels']:
        value=roots[channel['root_index']]
        momentum=np.sqrt(value)
        offset=q-value
        projector=.5*(np.eye(2)+(p.soc*momentum*c.fullk.SX+odd*c.fullk.SZ)/offset)
        velocity=2*momentum+p.soc**2*momentum/offset
        density+=momentum*projector/(2*math.pi*velocity*channel['velocity_sign'])
    return c.fullk.analytic_physical_self_energy_from_q(q,odd,p)-2j*math.pi*width**2*density


def channel_gate(config,count):
    if config.get('pole_sheet'):
        return count>=config['ep'].get('minimum_physical_open_channels',1)
    return count==config['ep']['required_open_channels']
