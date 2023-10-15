import argparse
import json

from functions import cluster_similar_elements  

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input")
    parser.add_argument("--output")
    parser.add_argument("--method",default='kmeans')
    parser.add_argument("--linkage_method", default='ward', help="Linkage method for hierarchical clustering")
    parser.add_argument("--distance_threshold", default=0.35, help="Distance threshold for forming clusters")
    parser.add_argument("--engine", default="text-embedding-ada-002", help="OpenAI engine for text embedding")
    args = parser.parse_args()

    if args.input is None:
        parser.error("Please provide the input file path.")
    else:
        input_file_path = args.input

    if args.output is None:
        with open(input_file_path, 'r') as input_file:
          input_data = json.load(input_file)

        input_data["titles"]={f"theme{i}": title for i, title in enumerate(input_data["titles"].values(), 1)}
        input_data["theme_attributes"]={f"theme{i}": title for i, title in enumerate(input_data["theme_attributes"].values(), 1)}
        
        cluster_similar_elements(input_data,method=args.method,distance_threshold=args.distance_threshold,engine=args.engine,linkage_method=args.linkage_method)
    else:
        output_file_path = args.output
        cluster_similar_elements(input_data,method=args.method,distance_threshold=args.distance_threshold,engine=args.engine,linkage_method=args.linkage_method)


    print("Clustering completed.")

if __name__ == "__main__":
    main()
#%%
