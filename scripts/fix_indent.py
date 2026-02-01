
lines_to_indent_start = 214
lines_to_indent_end = 526

file_path = "training/train.py"

with open(file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

new_lines = []
for i, line in enumerate(lines):
    line_num = i + 1
    if lines_to_indent_start <= line_num <= lines_to_indent_end:
        # Add 4 spaces if the line is not empty
        if line.strip():
            new_lines.append("    " + line)
        else:
            new_lines.append(line)
    else:
        new_lines.append(line)

with open(file_path, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("Indentation fixed.")
