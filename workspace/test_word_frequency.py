from word_frequency import word_frequency

# Create a sample text file
sample_text = "Hello, world! Hello... This is a test. A test this is."

sample_file = 'sample.txt'
with open(sample_file, 'w', encoding='utf-8') as f:
    f.write(sample_text)

# Call the function and print the result
result = word_frequency(sample_file)
print(result)
