import json

# Weights as defined in the problem
weights = {
    "w_s": 0.2,
    "w_description": 0.5,
    "w_comments": 0.3,
    "w_exe_d": 0.4,
    "w_ext_d": 0.3,
    "w_prop_d": 0.3,
    "w_exe_c": 0.2,
    "w_ext_c": 0.4,
    "w_prop_c": 0.4
}

def calculate_normalized_score(hit_score, max_hit_score):
    """Normalize the hit score using s_norm = s / s_max"""
    return hit_score / max_hit_score if max_hit_score > 0 else 0

def calculate_description_score(issue, weights):
    """Calculate the description score for an issue"""
    exe_d = issue["exe"]
    ext_d = issue["ext"]
    prop_d = issue["prop"]
    
    description_score = (weights["w_exe_d"] * exe_d) + \
                        (weights["w_ext_d"] * ext_d) + \
                        (weights["w_prop_d"] * prop_d)
    
    return weights["w_description"] * description_score

def calculate_comments_score(comments, weights):
    """Calculate the comments score using the average confidences"""
    if len(comments) == 0:
        return 0

    exe_C = sum([comment["exe_conf"] for comment in comments]) / len(comments)
    ext_C = sum([comment["ext_conf"] for comment in comments]) / len(comments)
    prop_C = sum([comment["prop_conf"] for comment in comments]) / len(comments)
    
    comments_score = (weights["w_exe_c"] * exe_C) + \
                     (weights["w_ext_c"] * ext_C) + \
                     (weights["w_prop_c"] * prop_C)
    
    return weights["w_comments"] * comments_score

def calculate_new_score(issue, comments, max_hit_score, weights):
    """Calculate the new score for an issue"""
    # Step 1: Normalize the hit score
    s_norm = calculate_normalized_score(issue["s_norm"], max_hit_score)
    
    # Step 2: Calculate the description score
    description_score = calculate_description_score(issue, weights)
    
    # Step 3: Calculate the comments score
    comments_score = calculate_comments_score(comments, weights)
    
    # Step 4: Combine to form the final score
    new_score = (weights["w_s"] * s_norm) + description_score + comments_score
    
    return new_score

def rank_issues(issues, comments, weights):
    """Rank issues based on the calculated new score"""
    # Find the maximum hit score to normalize
    max_hit_score = max(issue["s_norm"] for issue in issues)

    # Create a list of issues with their calculated scores
    ranked_issues = []
    for issue in issues:
        # Get the relevant comments for the issue
        issue_comments = [comment for comment in comments if comment["issue_id"] == issue["issue_id"]]
        
        # Calculate the new score for the issue
        new_score = calculate_new_score(issue, issue_comments, max_hit_score, weights)
        
        # Append the issue with its calculated score
        ranked_issues.append({
            "issue_id": issue["issue_id"],
            "new_score": new_score,
            "description": issue["description"]
        })

    # Sort issues by the new score in descending order
    ranked_issues.sort(key=lambda x: x["new_score"], reverse=True)
    
    return ranked_issues

def main():
    # Load the JSON data from file
    with open('issues_comments_dataset.json', 'r') as file:
        data = json.load(file)
    
    issues = data["issues"]
    comments = data["comments"]

    # Rank the issues
    ranked_issues = rank_issues(issues, comments, weights)

    # Output the ranked issues
    print("Ranked Issues (by new_score):")
    for rank, issue in enumerate(ranked_issues, start=1):
        print(f"Rank {rank}: Issue ID {issue['issue_id']} with score {issue['new_score']:.4f} - {issue['description']}")

if __name__ == "__main__":
    main()
