import importlib
import torch
import pytest
import rootutils
rootutils.setup_root(__file__, indicator=".project_root", pythonpath=True)

def test_module():
    assert True

@pytest.fixture
def get_input():
    return torch.rand(1, 3, 256, 256)

class TestWavemodule:
    module = importlib.import_module("ldm.modules.diffusionmodules.wavemodel")
    res = None

    @pytest.mark.parametrize(('level'),((1,2,3)))
    def test_encoder1(self, get_input, level):
        encoder = self.module.WaveEncoder(level=level)
        self.res = encoder(get_input)

    def test_decoder(self):
        decoder = self.module.WaveDecoder()
        o = decoder(self.res)
        assert o.shape == (1, 3, 256, 256)
