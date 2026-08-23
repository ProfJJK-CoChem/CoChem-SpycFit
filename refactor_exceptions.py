import re
import sys
from pathlib import Path

def process_file(filepath):
    path = Path(filepath)
    content = path.read_text(encoding='utf-8')
    
    # Replace all broad exceptions that just log debug/warning notices with specific ones or remove them if possible.
    # Since these are mostly IO / lock operations, OSError is the most likely intended exception.
    
    # Example: except Exception as err:\n logger.debug(...)
    # Let's replace 'except Exception as' with 'except (OSError, ValueError, TypeError) as'
    
    pattern1 = re.compile(r'except Exception as (\w+):\s+logger\.debug\(')
    content = pattern1.sub(r'except (OSError, ValueError, TypeError) as \1:\n                logger.debug(', content)
    
    pattern2 = re.compile(r'except Exception as (\w+):\s+logger\.warning\(')
    content = pattern2.sub(r'except (OSError, ValueError, TypeError) as \1:\n                logger.warning(', content)
    
    # Let's also look for generic exceptions in cochem_spycfit_init.py
    
    path.write_text(content, encoding='utf-8')

if __name__ == '__main__':
    process_file(r'D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\core_engine\cochem_fit_registry_manager.py')
    process_file(r'D:\__CoChem\GitHub-Repo\CoChem-SpycFit\src\cochem_spycfit\core_engine\cochem_spycfit_init.py')
    print("Done")
