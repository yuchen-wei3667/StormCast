import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
import math
from StormCast.forecast import forecast_with_uncertainty
import StormCast.forecast

def test_radius(scale_val):
    with open('/home/yuchenwei/Projects/StormCast/src/StormCast/forecast.py', 'r') as f:
        content = f.read()
    
    import re
    new_content = re.sub(r'radius = max\(sigma_pos\[0\], sigma_pos\[1\]\) \* math\.sqrt\([^\)]+\)', 
                         f'radius = max(sigma_pos[0], sigma_pos[1]) * math.sqrt({scale_val})', content)
    
    with open('/home/yuchenwei/Projects/StormCast/src/StormCast/forecast.py', 'w') as f:
        f.write(new_content)
        
    print(f"Testing scale_val = {scale_val}")
    os.system("python scripts/evaluate_hit_rate_by_history.py > tmp_results.txt")
    os.system("cat tmp_results.txt | grep 'min |'")
    
test_radius(2.30)
