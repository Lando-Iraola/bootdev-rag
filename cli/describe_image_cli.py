import argparse

from lib.describe_image import describe_image


def main(rag_summarize=None):
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")



    parser.add_argument("--image",required=True, type=str, help="Path to an image")
    parser.add_argument("--query", required=True, type=str, help="a text query to rewrite based on the image")

    args = parser.parse_args()

    response = describe_image(args.image, args.query)
    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")



if __name__ == "__main__":
    main()
