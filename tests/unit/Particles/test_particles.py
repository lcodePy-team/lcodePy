import os
import tempfile
import pytest
import numpy as np
from typing_extensions import Annotated

from lcode.beam.data import Particles, BeamParticles

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None


@pytest.fixture
def TestParticle():
    """Fixture for test particle class"""
    class TestParticle(Particles):
        x: Annotated[np.ndarray, 'f8']
        y: Annotated[np.ndarray, 'f8']
        flag: Annotated[np.ndarray, '?', 'extra']
    
    return TestParticle


class TestParticles:
    
    def test_init_empty(self, TestParticle):
        """Test creating empty particle object"""
        p = TestParticle()
        assert p.size == 0
        assert len(p.x) == 0
        assert len(p.y) == 0
        assert len(p.flag) == 0
    
    def test_init_from_structured_array(self, TestParticle):
        """Test creation from structured array"""
        data = np.array([(1.0, 2.0), (3.0, 4.0)], 
                       dtype=[('x', 'f8'), ('y', 'f8')])
        p = TestParticle(beam_array=data)
        
        assert p.size == 2
        np.testing.assert_array_equal(p.x, [1.0, 3.0])
        np.testing.assert_array_equal(p.y, [2.0, 4.0])
        np.testing.assert_array_equal(p.flag, [False, False])
    
    def test_init_from_regular_array(self, TestParticle):
        """Test creation from regular numpy array"""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        p = TestParticle(beam_array=data)
        
        assert p.size == 2
        np.testing.assert_array_equal(p.x, [1.0, 3.0])
        np.testing.assert_array_equal(p.y, [2.0, 4.0])
    
    def test_getitem_string(self, TestParticle):
        """Test field access by name"""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        p = TestParticle(beam_array=data)
        
        np.testing.assert_array_equal(p['x'], [1.0, 3.0])
        np.testing.assert_array_equal(p['y'], [2.0, 4.0])
    
    def test_getitem_slice(self, TestParticle):
        """Test getting particle slice"""
        data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        p = TestParticle(beam_array=data)
        
        sliced = p[1:]
        assert sliced.size == 2
        np.testing.assert_array_equal(sliced.x, [3.0, 5.0])
    
    def test_setitem_string(self, TestParticle):
        """Test setting field by name"""
        p = TestParticle()
        with pytest.raises(ValueError):
            p['x'] = np.array([1.0, 2.0])
    
    def test_append(self, TestParticle):
        """Test appending particles"""
        data1 = np.array([[1.0, 2.0]])
        data2 = np.array([[3.0, 4.0]])
        
        p1 = TestParticle(beam_array=data1)
        p2 = TestParticle(beam_array=data2)
        
        p1.append(p2)
        
        assert p1.size == 2
        np.testing.assert_array_equal(p1.x, [1.0, 3.0])
    
    def test_sort(self, TestParticle):
        """Test particle sorting"""
        data = np.array([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]])
        p = TestParticle(beam_array=data)
        
        p.sort(p.x)
        
        np.testing.assert_array_equal(p.x, [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(p.y, [3.0, 2.0, 1.0])
    
    def test_particles_property(self, TestParticle):
        """Test getting particle matrix"""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        p = TestParticle(beam_array=data)
        
        particles = p.as_array()
        expected = np.array([[1.0, 2.0], [3.0, 4.0]])
        np.testing.assert_array_equal(particles, expected)
    
    def test_save_load(self, TestParticle, tmp_path):
        """Test save and load functionality"""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        p1 = TestParticle(beam_array=data)
        
        temp_file = tmp_path / "test_particles.npz"
        
        p1.save(str(temp_file))
        
        p2 = TestParticle()
        p2.load(str(temp_file))
        
        assert p2.size == 2
        np.testing.assert_array_equal(p2.x, p1.x)
        np.testing.assert_array_equal(p2.y, p1.y)


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
class TestParticlesCuPy:
    """Test Particles class with CuPy arrays"""
    
    @pytest.fixture
    def TestParticleCuPy(self):
        """Fixture for CuPy test particle class"""
        class TestParticle(Particles):
            x: Annotated[np.ndarray, 'f8']
            y: Annotated[np.ndarray, 'f8']
            flag: Annotated[np.ndarray, '?', 'extra']
        
        return TestParticle
    
    def test_init_empty_cupy(self, TestParticleCuPy):
        """Test creating empty particle object with CuPy"""
        p = TestParticleCuPy(xp=cp)
        assert p.size == 0
        assert len(p.x) == 0
        assert isinstance(p.x, cp.ndarray)
    
    def test_init_from_array_cupy(self, TestParticleCuPy):
        """Test creation from CuPy array"""
        data = cp.array([[1.0, 2.0], [3.0, 4.0]])
        p = TestParticleCuPy(xp=cp, beam_array=data)
        
        assert p.size == 2
        cp.testing.assert_array_equal(p.x, cp.array([1.0, 3.0]))
        cp.testing.assert_array_equal(p.y, cp.array([2.0, 4.0]))
    
    def test_getitem_slice_cupy(self, TestParticleCuPy):
        """Test getting particle slice with CuPy"""
        data = cp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        p = TestParticleCuPy(xp=cp, beam_array=data)
        
        sliced = p[1:]
        assert sliced.size == 2
        cp.testing.assert_array_equal(sliced.x, cp.array([3.0, 5.0]))
    
    def test_append_cupy(self, TestParticleCuPy):
        """Test appending particles with CuPy"""
        data1 = cp.array([[1.0, 2.0]])
        data2 = cp.array([[3.0, 4.0]])
        
        p1 = TestParticleCuPy(xp=cp, beam_array=data1)
        p2 = TestParticleCuPy(xp=cp, beam_array=data2)
        
        p1.append(p2)
        
        assert p1.size == 2
        cp.testing.assert_array_equal(p1.x, cp.array([1.0, 3.0]))
    
    def test_sort_cupy(self, TestParticleCuPy):
        """Test particle sorting with CuPy"""
        data = cp.array([[3.0, 1.0], [1.0, 3.0], [2.0, 2.0]])
        p = TestParticleCuPy(xp=cp, beam_array=data)
        
        p.sort(p.x)
        
        cp.testing.assert_array_equal(p.x, cp.array([1.0, 2.0, 3.0]))
        cp.testing.assert_array_equal(p.y, cp.array([3.0, 2.0, 1.0]))
    
    def test_particles_property_cupy(self, TestParticleCuPy):
        """Test getting particle matrix with CuPy"""
        data = cp.array([[1.0, 2.0], [3.0, 4.0]])
        p = TestParticleCuPy(xp=cp, beam_array=data)
        
        particles = p.as_array()
        expected = cp.array([[1.0, 2.0], [3.0, 4.0]])
        cp.testing.assert_array_equal(particles, expected)


@pytest.mark.skipif(not HAS_CUPY, reason="CuPy not available")
class TestBeamParticlesCuPy:
    """Test BeamParticles class with CuPy arrays"""
    
    def test_init_from_array_cupy(self):
        """Test beam creation from CuPy array"""
        data = cp.zeros((10, 8))
        data[:, 0] = cp.arange(10)  # xi
        
        beam = BeamParticles(xp=cp, beam_array=data)
        assert beam.size == 10
        cp.testing.assert_array_equal(beam.xi, cp.arange(10))
    
    def test_sort_by_xi_cupy(self):
        """Test sorting by xi with CuPy"""
        data = cp.zeros((5, 8))
        data[:, 0] = cp.array([5, 2, 8, 1, 3])  # xi
        
        beam = BeamParticles(xp=cp, beam_array=data)
        beam.sort_by_xi()
        
        # Sort in descending order (-xi)
        cp.testing.assert_array_equal(beam.xi, cp.array([8, 5, 3, 2, 1]))
    
    def test_cut_beam_layer_cupy(self):
        """Test cutting beam layer with CuPy"""
        data = cp.zeros((10, 8))
        data[:, 0] = cp.arange(10)
        
        beam = BeamParticles(xp=cp, beam_array=data)
        layer, remaining = beam.cut_beam_layer(3)
        
        assert layer.size == 3
        assert remaining.size == 7
        cp.testing.assert_array_equal(layer.xi, cp.array([0, 1, 2]))
        cp.testing.assert_array_equal(remaining.xi, cp.arange(3, 10))
    
    def test_nlost_property_cupy(self):
        """Test counting lost particles with CuPy"""
        data = cp.zeros((5, 8))
        beam = BeamParticles(xp=cp, beam_array=data)
        
        beam.lost = cp.array([True, False, True, False, False])
        
        assert int(beam.nlost) == 2