from flask import Flask, request, jsonify
from services.input import handle_input
from services.output import generate_output

app = Flask(__name__)

@app.route('/api/input', methods=['POST'])
def handle_input():
    input_data = request.json['input']
    handle_input(input_data)
    output_data = generate_output(input_data)
    return jsonify({'output': output_data})

if __name__ == '__main__':
    app.run(debug=True)