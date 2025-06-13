# Invoice Reader using Donut

This project fine-tunes the Donut (Large Language and Vision Assistant) model to extract information from invoices. The model is trained to understand and extract key information from invoice images.

## Project Structure

```
invoice_reader/
├── data/                 # Directory for invoice images and annotations
│   ├── raw/              # Raw invoice images
│   └── processed/        # Processed and annotated data
├── src/                  # Source code
│   ├── data/             # Data processing scripts
│   ├── model/            # Model training and inference code
│   └── utils/            # Utility functions
├── notebooks/            # Jupyter notebooks for experimentation
├── configs/              # Configuration files
├── requirements.txt      # Project dependencies
└── README.md             # Project documentation
```

## .gitignore

This project includes a `.gitignore` file to exclude unnecessary files and directories from version control, such as virtual environments, logs, and temporary files. Note that the `data/` directory is tracked by git, so any data placed here will be included in version control unless otherwise specified. If you wish to ignore the `data/` directory, you can uncomment the relevant line in `.gitignore`.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

[To be added as the project develops]

## License

[Add your chosen license]
