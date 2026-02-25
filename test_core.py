import unittest
import numpy as np
import core

class TestUAVEnergyModelFramework(unittest.TestCase):
    
    def test_feasibility_logic(self):
        """Test feasibility filtering based on paper rules."""
        # Rule: O1 admissible if F4 = 1
        # F-vector: [F1, F2, F3, F4, F5, F6] -> indices 0..5
        # F4 is index 3.
        f_vector = [0, 0, 0, 1, 0, 0] 
        admissible = core.check_feasibility(f_vector)
        self.assertIn(1, admissible, "O1 should be admissible when F4=1")
        
        # Rule: O2 admissible if F2 = 1 (index 1)
        f_vector = [0, 1, 0, 0, 0, 0]
        admissible = core.check_feasibility(f_vector)
        self.assertIn(2, admissible, "O2 should be admissible when F2=1")
        
        # Rule: O3 admissible if F5 = 1 and F6 = 0 (indices 4 and 5)
        f_vector = [0, 0, 0, 0, 1, 0]
        admissible = core.check_feasibility(f_vector)
        self.assertIn(3, admissible, "O3 should be admissible when F5=1, F6=0")
        
        # Rule: O4 admissible if F5 = 1 and F6 = 1
        f_vector = [0, 0, 0, 0, 1, 1]
        admissible = core.check_feasibility(f_vector)
        self.assertIn(4, admissible, "O4 should be admissible when F5=1, F6=1")
        self.assertNotIn(3, admissible, "O3 should NOT be admissible when F6=1")
        
        # Rule: O5 admissible if F3 = 1 (index 2)
        f_vector = [0, 0, 1, 0, 0, 0]
        admissible = core.check_feasibility(f_vector)
        self.assertIn(5, admissible, "O5 should be admissible when F3=1")

    def test_local_priority_vectors(self):
        """Verify hardcoded local priority vectors match paper values."""
        # Paper values
        # A1 (Accuracy): [0.38, 0.27, 0.08, 0.14, 0.13]
        expected_a1 = np.array([0.38, 0.27, 0.08, 0.14, 0.13])
        np.testing.assert_allclose(core.LOCAL_PRIORITIES[1], expected_a1, atol=1e-2)
        
        # A2 (Interpretability): [0.42, 0.30, 0.08, 0.05, 0.15]
        expected_a2 = np.array([0.42, 0.30, 0.08, 0.05, 0.15])
        np.testing.assert_allclose(core.LOCAL_PRIORITIES[2], expected_a2, atol=1e-2)

    def test_global_aggregation_example(self):
        """Verify the 'Numerical Example' from the paper."""
        # Paper Example Weights
        # w = [0.40, 0.30, 0.20, 0.10]
        weights = np.array([0.40, 0.30, 0.20, 0.10])
        
        # Calculate scores manually using core's LOCAL_PRIORITIES
        S = np.zeros(5)
        for i in range(4):
            S += weights[i] * core.LOCAL_PRIORITIES[i+1]
            
        # Expected Scores from Paper
        # O1: 0.329
        # O2: 0.245
        # O3: 0.099
        # O4: 0.141
        # O5: 0.186
        expected_scores = np.array([0.329, 0.245, 0.099, 0.141, 0.186])
        
        print("\nTest Global Aggregation Results:")
        print(f"Calculated: {S}")
        print(f"Expected:   {expected_scores}")
        
        np.testing.assert_allclose(S, expected_scores, atol=1e-3, 
                                   err_msg="Global scores do not match paper example")

    def test_consistency_calculation(self):
        """Test CI/CR calculation logic."""
        # Perfectly consistent matrix (3x3)
        # 1  2  4
        # .5 1  2
        # .25 .5 1
        M = np.array([
            [1, 2, 4],
            [0.5, 1, 2],
            [0.25, 0.5, 1]
        ])
        vec, cr = core.calculate_priority_vector(M)
        self.assertLess(cr, 1e-5, "Perfectly consistent matrix should have CR ~ 0")
        
        # Known inconsistent matrix
        # 1 9 9
        # 1/9 1 9
        # 1/9 1/9 1
        M_incons = np.array([
            [1, 9, 9],
            [1/9, 1, 9],
            [1/9, 1/9, 1]
        ])
        vec, cr = core.calculate_priority_vector(M_incons)
        self.assertGreater(cr, 0.1, "Inconsistent matrix should have CR > 0.1")

if __name__ == '__main__':
    unittest.main()
