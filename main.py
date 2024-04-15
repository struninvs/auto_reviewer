from streamlit.web.cli import main
import sys

if __name__ == "__main__":
    # Set prog_name so that the Streamlit server sees the same command line
    # string whether streamlit is called directly or via `python -m streamlit`.
    sys.argv = ["streamlit", "run", "reviews_helper.py", ""]
    sys.exit(main(prog_name="streamlit"))