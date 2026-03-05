import pytest
import numpy as np

from lcode.beam.data import BeamParticles as BeamParticles2D
from lcode.beam3d.data import BeamParticles as BeamParticles3D

class TestBeamParticles2D:
    
    def test_init_empty(self):
        """Test creating empty beam"""
        beam = BeamParticles2D()
        assert beam.size == 0
        beam.xi, beam.r, beam.p_z
        beam.p_r, beam.M, beam.q_m
        beam.q_norm, beam.id
        beam.dt, beam.remaining_steps
        beam.lost
    
    def test_init_from_array(self):
        """Test beam creation from array"""
        data = np.zeros((10, 8))
        data[:, 0] = np.arange(10)  # xi
        
        beam = BeamParticles2D(beam_array=data)
        assert beam.size == 10
        np.testing.assert_array_equal(beam.xi, np.arange(10))
    
    def test_sort_by_xi(self):
        """Test sorting by xi coordinate"""
        data = np.zeros((5, 8))
        data[:, 0] = [5, 2, 8, 1, 3]  # xi
        
        beam = BeamParticles2D(beam_array=data)
        beam.sort_by_xi()
        
        # Sort in descending order (-xi)
        np.testing.assert_array_equal(beam.xi, [8, 5, 3, 2, 1])
    
    def test_cut_beam_layer(self):
        """Test cutting beam layer"""
        data = np.zeros((10, 8))
        data[:, 0] = np.arange(10)
        
        beam = BeamParticles2D(beam_array=data)
        layer, remaining = beam.cut_beam_layer(3)
        
        assert layer.size == 3
        assert remaining.size == 7
        np.testing.assert_array_equal(layer.xi, [0, 1, 2])
        np.testing.assert_array_equal(remaining.xi, np.arange(3, 10))
    
    def test_nlost_property(self):
        """Test counting lost particles"""
        data = np.zeros((5, 8))
        beam = BeamParticles2D(beam_array=data)
        
        beam.lost = np.array([True, False, True, False, False])
        
        assert beam.nlost == 2
    
    def test_extra_dtype_initialization(self):
        """Test extra dtype field initialization"""
        data = np.zeros((3, 8))
        beam = BeamParticles2D(beam_array=data)
        
        assert len(beam.dt) == 3
        assert len(beam.remaining_steps) == 3
        assert len(beam.lost) == 3
        
        # Check initial values
        np.testing.assert_array_equal(beam.dt, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(beam.lost, [False, False, False])
    
    @pytest.mark.parametrize("layer_size,expected_layer,expected_remaining", [
        (0, 0, 10),
        (5, 5, 5),
        (10, 10, 0),
    ])
    def test_cut_beam_layer_parametrized(self, layer_size, expected_layer, expected_remaining):
        """Parametrized test for beam layer cutting"""
        data = np.zeros((10, 8))
        beam = BeamParticles2D(beam_array=data)
        
        layer, remaining = beam.cut_beam_layer(layer_size)
        
        assert layer.size == expected_layer
        assert remaining.size == expected_remaining


class TestBeamParticles3D:
    
    def test_init_empty(self):
        """Test creating empty beam"""
        beam = BeamParticles3D()
        assert beam.size == 0
        beam.xi, beam.x, beam.y
        beam.px, beam.py, beam.pz
        beam.q_m, beam.q_norm, beam.id
        beam.dt, beam.remaining_steps
    
    def test_init_from_array(self):
        """Test beam creation from array"""
        data = np.zeros((10, 9))
        data[:, 0] = np.arange(10)  # xi
        
        beam = BeamParticles3D(beam_array=data)
        assert beam.size == 10
        np.testing.assert_array_equal(beam.xi, np.arange(10))
    
    def test_sort_by_xi(self):
        """Test sorting by xi coordinate"""
        data = np.zeros((5, 9))
        data[:, 0] = [5, 2, 8, 1, 3]  # xi
        
        beam = BeamParticles3D(beam_array=data)
        beam.sort_by_xi()
        
        # Sort in descending order (-xi)
        np.testing.assert_array_equal(beam.xi, [8, 5, 3, 2, 1])
    
    def test_cut_beam_layer(self):
        """Test cutting beam layer"""
        data = np.zeros((10, 9))
        data[:, 0] = np.arange(10)
        
        beam = BeamParticles3D(beam_array=data)
        layer, remaining = beam.cut_beam_layer(3)
        
        assert layer.size == 3
        assert remaining.size == 7
        np.testing.assert_array_equal(layer.xi, [0, 1, 2])
        np.testing.assert_array_equal(remaining.xi, np.arange(3, 10))
    
    def test_extra_dtype_initialization(self):
        """Test extra dtype field initialization"""
        data = np.zeros((3, 9))
        beam = BeamParticles3D(beam_array=data)
        
        assert len(beam.dt) == 3
        assert len(beam.remaining_steps) == 3
        
        # Check initial values
        np.testing.assert_array_equal(beam.dt, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(beam.remaining_steps, [0.0, 0.0, 0.0])

    
    @pytest.mark.parametrize("layer_size,expected_layer,expected_remaining", [
        (0, 0, 10),
        (5, 5, 5),
        (10, 10, 0),
    ])
    def test_cut_beam_layer_parametrized(self, layer_size, expected_layer, expected_remaining):
        """Parametrized test for beam layer cutting"""
        data = np.zeros((10, 9))
        beam = BeamParticles3D(beam_array=data)
        
        layer, remaining = beam.cut_beam_layer(layer_size)
        
        assert layer.size == expected_layer
        assert remaining.size == expected_remaining