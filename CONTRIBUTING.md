# Contributing to Collembola Detection Pipeline

Thank you for your interest in contributing to the Collembola Detection Pipeline! This document provides guidelines for contributing to the project.

## How to Contribute

### Reporting Issues

If you encounter bugs or have feature requests:

1. Check the [existing issues](https://github.com/QuantEcoLab/collembolae_vis/issues) to avoid duplicates
2. Create a new issue with:
   - Clear, descriptive title
   - Detailed description of the problem or feature
   - Steps to reproduce (for bugs)
   - Expected vs actual behavior
   - Your environment (OS, Python version, GPU/CPU)

### Pull Requests

We welcome pull requests! To contribute code:

1. **Fork the repository** and create a new branch from `main`
2. **Make your changes** following our code style
3. **Test your changes** thoroughly
4. **Update documentation** if needed
5. **Submit a pull request** with:
   - Clear description of changes
   - Link to related issue(s)
   - Test results and validation

### Code Style

- Follow PEP 8 Python style guidelines
- Use meaningful variable and function names
- Add docstrings to functions and classes
- Comment complex logic
- Keep functions focused and modular

### Testing

Before submitting:
- Test your code on sample data
- Verify backward compatibility
- Check that existing scripts still work
- Document any new dependencies

## Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/collembolae_vis.git
cd collembolae_vis

# Create conda environment
conda create -n collembola python=3.11
conda activate collembola

# Install dependencies
pip install -r requirements.txt
```

## Project Areas for Contribution

### High Priority
- Instance segmentation implementation (YOLO-seg)
- Web interface for batch processing
- Automated quality control and validation
- Performance optimizations
- Additional measurement methods

### Documentation
- Tutorial notebooks
- Use case examples
- API documentation
- Video tutorials

### Testing
- Unit tests for core functions
- Integration tests
- Performance benchmarks
- Edge case handling

## Questions?

Feel free to:
- Open an issue for questions
- Start a discussion in GitHub Discussions
- Contact the maintainers

## Code of Conduct

Please be respectful and constructive in all interactions. We aim to create a welcoming environment for all contributors.

---

Thank you for helping improve the Collembola Detection Pipeline!
