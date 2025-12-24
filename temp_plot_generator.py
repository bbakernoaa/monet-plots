import os
import re
import subprocess

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend to prevent plots from displaying


def generate_gallery_images():
    """
    Generate images for the example gallery by executing Python code blocks
    in the markdown files and saving the plots to the correct directory.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    examples_dir = os.path.join(project_root, "docs", "examples", "getting-started")

    for filename in os.listdir(examples_dir):
        if filename.endswith(".md"):
            filepath = os.path.join(examples_dir, filename)
            base_filename, _ = os.path.splitext(filename)
            output_image_path = os.path.join(examples_dir, f"{base_filename}.png")

            with open(filepath, "r") as f:
                content = f.read()

            # Find all python code blocks
            python_code_blocks = re.findall(r"```python(.*?)```", content, re.DOTALL)

            if python_code_blocks:
                # Combine all code blocks into a single script
                full_python_code = "\n".join(python_code_blocks)

                # Replace plt.show() with plt.savefig() to save the plot
                full_python_code = full_python_code.replace("plt.show()", f"plt.savefig('{output_image_path}')")

                # Create a temporary script to execute
                script_path = "temp_script.py"
                with open(script_path, "w") as temp_script:
                    temp_script.write(full_python_code)

                # Execute the script
                print(f"Generating plot for {filename}...")
                try:
                    subprocess.run(
                        ["python", script_path],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                    print(f"Successfully generated {output_image_path}")
                except subprocess.CalledProcessError as e:
                    print(f"Error generating plot for {filename}:")
                    print(e.stdout)
                    print(e.stderr)

                # Clean up the temporary script
                os.remove(script_path)


if __name__ == "__main__":
    generate_gallery_images()
