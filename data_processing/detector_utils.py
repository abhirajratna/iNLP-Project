import re
import numpy as np

class CodeStyler:
    """Extracts stylistic features from code to distinguish between Human and AI authors."""
    
    @staticmethod
    def extract_features(code: str):
        if not isinstance(code, str) or not code.strip():
            return {
                'comment_count': 0,
                'snake_case_count': 0,
                'camel_case_count': 0,
                'include_count': 0,
                'avg_line_length': 0,
                'total_lines': 0
            }
        
        lines = code.split('\n')
        total_lines = len(lines)
        
        # 1. Comment density
        comment_count = len(re.findall(r'//|/\*', code))
        
        # 2. Identifier style
        # snake_case: lowercase_lowercase
        snake_case_count = len(re.findall(r'[a-z][a-z0-9]*_[a-z0-9_]+', code))
        # camelCase: lowercaseUPPERCASE
        camel_case_count = len(re.findall(r'[a-z][a-z0-9]*[A-Z][a-zA-Z0-9]*', code))
        
        # 3. Boilerplate
        include_count = len(re.findall(r'#include', code))
        
        # 4. Length
        avg_line_length = np.mean([len(l) for l in lines]) if lines else 0
        
        return {
            'comment_density': comment_count / max(total_lines, 1),
            'snake_case_count': snake_case_count,
            'camel_case_count': camel_case_count,
            'include_count': include_count,
            'avg_line_length': avg_line_length,
            'total_lines': total_lines
        }

    @staticmethod
    def classify(code: str):
        """
        Classifies code as 'Human' or 'AI' based on learned heuristics from ai_hum.csv.
        Returns (label, confidence_score)
        """
        feats = CodeStyler.extract_features(code)
        
        # Heuristic scoring (positive = Human, negative = AI)
        score = 0
        
        # 1. Comments: Humans write significantly more comments
        if feats['comment_density'] > 0.2:
            score += 2
        elif feats['comment_density'] == 0:
            score -= 1
            
        # 2. snake_case: Strong human indicator in this dataset
        if feats['snake_case_count'] > feats['camel_case_count'] * 1.5:
            score += 2
        elif feats['camel_case_count'] > feats['snake_case_count']:
            score -= 1
            
        # 3. Line length: AI tends to have longer lines
        if feats['avg_line_length'] > 30:
            score -= 2
        elif feats['avg_line_length'] < 20:
            score += 1
            
        # 4. Includes: AI often has more boilerplate
        if feats['include_count'] > 4:
            score -= 1
            
        # Final decision
        if score >= 1:
            label = "Human"
            confidence = min(abs(score) / 5.0, 1.0)
        else:
            label = "AI"
            confidence = min(abs(score) / 5.0, 1.0)
            
        return label, confidence

if __name__ == "__main__":
    # Small test
    sample_ai = "#include <iostream>\nint main() { std::cout << \"Hello World\"; return 0; }"
    sample_hum = "// My main function\nint main_loop() {\n  int max_val = 10; // set max\n  return 0;\n}"
    
    print(f"AI Sample Result: {CodeStyler.classify(sample_ai)}")
    print(f"Human Sample Result: {CodeStyler.classify(sample_hum)}")
