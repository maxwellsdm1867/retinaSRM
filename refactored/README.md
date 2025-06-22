# Refactored Sodium Channel Dynamics Analysis

This directory contains a refactored version of the sodium channel dynamics analysis code, organized into a modular, maintainable structure.

## Overview

The refactored code addresses the original code's complexity by:

1. **Modular Design**: Breaking down the monolithic script into focused, single-responsibility classes
2. **Clear Separation of Concerns**: Separating configuration, data processing, analysis, and reporting
3. **Improved Error Handling**: Comprehensive error handling and logging throughout the pipeline
4. **Better Documentation**: Detailed documentation for all classes and methods
5. **Enhanced Maintainability**: Easier to modify, extend, and debug individual components

## Directory Structure

```
refactored/
├── ai_population_analysis_main.m    # Main orchestrator script
├── config/                          # Configuration classes
│   ├── AnalysisConfig.m            # Analysis parameters and settings
│   ├── PathConfig.m                # File paths and directories
│   └── SpikeConfig.m               # Spike detection parameters
├── core/                           # Core functionality modules
│   ├── TreeNavigator.m             # Tree traversal and node selection
│   ├── DataExtractor.m             # Data extraction from nodes
│   └── AnalysisExecutor.m          # Core analysis logic
├── analysis/                       # Analysis modules
│   └── FrequencyAnalyzer.m         # Frequency-specific analysis
├── reporting/                      # Reporting modules
│   ├── AnalysisReporter.m          # Analysis reports
│   ├── DebugReporter.m             # Debug reports and logs
│   └── StatisticsReporter.m        # Statistical reports
└── README.md                       # This file
```

## Key Components

### 1. Main Orchestrator (`ai_population_analysis_main.m`)

The main entry point that coordinates the entire analysis pipeline:

- **Initialization**: Sets up environment, loads configurations, and validates settings
- **Tree Navigation**: Uses `TreeNavigator` to extract frequency nodes
- **Analysis Execution**: Orchestrates analysis using `AnalysisExecutor`
- **Reporting**: Generates comprehensive reports using specialized reporters
- **Error Handling**: Comprehensive error handling and recovery

### 2. Configuration Classes (`config/`)

#### `AnalysisConfig.m`
- Analysis parameters (sampling rates, filter settings, analysis windows)
- Parameter validation and utility methods
- Centralized configuration management

#### `PathConfig.m`
- File path management for data, logs, figures, and analysis results
- Automatic directory creation and validation
- Path utility methods

#### `SpikeConfig.m`
- Spike detection parameters (thresholds, search windows, quality criteria)
- Quality assessment methods
- Parameter validation and conversion utilities

### 3. Core Functionality (`core/`)

#### `TreeNavigator.m`
- Navigates the epoch tree structure
- Extracts frequency nodes and metadata
- Provides tree traversal utilities
- Handles node selection and validation

#### `DataExtractor.m`
- Extracts voltage and current data from nodes
- Performs signal preprocessing and cleaning
- Implements spike detection algorithms
- Computes spike-triggered averages
- Validates data quality

#### `AnalysisExecutor.m`
- Orchestrates the complete analysis pipeline
- Manages data flow between components
- Performs quality assessments
- Compiles and stores results
- Handles analysis failures gracefully

### 4. Analysis Modules (`analysis/`)

#### `FrequencyAnalyzer.m`
- Performs frequency-specific analysis
- Extracts frequency-dependent parameters
- Compares responses across frequencies
- Generates frequency-dependent statistics
- Identifies trends and patterns

### 5. Reporting Modules (`reporting/`)

#### `AnalysisReporter.m`
- Generates comprehensive analysis reports
- Computes summary statistics
- Analyzes quality metrics
- Provides recommendations
- Exports results to various formats

#### `DebugReporter.m`
- Manages debug logging throughout the pipeline
- Tracks execution flow and performance
- Provides diagnostic tools
- Exports debug information
- Supports filtering and analysis of debug data

#### `StatisticsReporter.m`
- Performs statistical analysis of results
- Computes descriptive statistics
- Analyzes distributions and trends
- Performs correlation analysis
- Generates statistical visualizations

## Usage

### Basic Usage

1. **Setup**: Ensure all dependencies are available and paths are configured
2. **Configuration**: Modify configuration files as needed for your analysis
3. **Execution**: Run the main script:
   ```matlab
   ai_population_analysis_main
   ```

### Advanced Usage

#### Custom Configuration
```matlab
% Create custom analysis configuration
analysis_config = AnalysisConfig();
analysis_config.cutoff_freq_Hz = 100;  % Custom filter cutoff
analysis_config.validate();            % Validate parameters

% Create custom spike detection configuration
spike_config = SpikeConfig();
spike_config.vm_thresh = -25;          % Custom voltage threshold
spike_config.validate();               % Validate parameters
```

#### Selective Analysis
```matlab
% Analyze specific frequency nodes
tree_navigator = TreeNavigator(rootNode);
[frequencyNodes, metadata] = tree_navigator.getAllFrequencyNodes();

% Analyze only specific nodes
for i = 1:length(frequencyNodes)
    if strcmp(frequencyNodes(i).cell, 'target_cell')
        result = analysis_executor.analyzeFrequencyNode(frequencyNodes(i));
    end
end
```

#### Custom Reporting
```matlab
% Generate custom reports
analysis_reporter = AnalysisReporter(analysis_results);
analysis_reporter.generateSummary();

% Generate debug report
debug_reporter = DebugReporter();
debug_reporter.generateDebugLog();

% Generate statistical report
stats_reporter = StatisticsReporter(analysis_results);
stats_reporter.generateStatistics();
```

## Key Improvements

### 1. Modularity
- **Single Responsibility**: Each class has a focused, well-defined purpose
- **Loose Coupling**: Classes interact through well-defined interfaces
- **High Cohesion**: Related functionality is grouped together

### 2. Error Handling
- **Comprehensive Error Handling**: Try-catch blocks around critical operations
- **Graceful Degradation**: System continues operation even when individual analyses fail
- **Detailed Error Reporting**: Rich error information for debugging

### 3. Logging and Debugging
- **Structured Logging**: Consistent log format with levels and categories
- **Debug Mode**: Optional detailed logging for troubleshooting
- **Performance Tracking**: Monitor execution time and resource usage

### 4. Configuration Management
- **Centralized Configuration**: All parameters in dedicated configuration classes
- **Parameter Validation**: Automatic validation of configuration parameters
- **Flexible Configuration**: Easy to modify parameters without code changes

### 5. Data Flow
- **Clear Data Flow**: Well-defined data flow between components
- **Data Validation**: Comprehensive validation at each step
- **Result Storage**: Structured storage of results for later retrieval

### 6. Extensibility
- **Plugin Architecture**: Easy to add new analysis modules
- **Customizable Reporting**: Flexible reporting system
- **Configuration Extensions**: Easy to add new configuration parameters

## Migration from Original Code

### Key Changes

1. **Function Extraction**: All inline functions moved to appropriate utility classes
2. **Loop Simplification**: Complex nested loops broken into focused methods
3. **Error Handling**: Added comprehensive error handling throughout
4. **Logging**: Implemented structured logging system
5. **Configuration**: Centralized all parameters in configuration classes

### Compatibility

- **Data Format**: Maintains compatibility with original data structures
- **Results Format**: Preserves original result format for backward compatibility
- **External Dependencies**: Uses same external functions and libraries

### Migration Steps

1. **Backup**: Create backup of original code
2. **Configuration**: Set up configuration files for your environment
3. **Testing**: Test with small dataset first
4. **Validation**: Compare results with original implementation
5. **Deployment**: Replace original code with refactored version

## Dependencies

### Required MATLAB Toolboxes
- Signal Processing Toolbox
- Statistics and Machine Learning Toolbox
- Image Processing Toolbox (for some visualizations)

### External Functions
- `detect_spike_initiation_elbow_v2.m`
- `estimate_filter_fft_trials_regularized.m`
- `getSelectedData.m`
- `getNoiseStm.m`
- `spkd_c.mexmaci64`

### Utility Classes
- `LoggingUtils.m`
- `ClipboardUtils.m`
- `DebugUtils.m`
- `VPLossUtils.m`

## Performance Considerations

### Optimization Features
- **Efficient Data Processing**: Optimized data extraction and processing
- **Memory Management**: Careful memory management for large datasets
- **Parallel Processing**: Support for parallel processing where applicable
- **Caching**: Intelligent caching of intermediate results

### Scalability
- **Modular Design**: Easy to scale individual components
- **Resource Management**: Efficient resource usage
- **Batch Processing**: Support for batch processing of multiple datasets

## Troubleshooting

### Common Issues

1. **Configuration Errors**: Check parameter validation in configuration classes
2. **Data Extraction Failures**: Verify data format and node structure
3. **Memory Issues**: Monitor memory usage with large datasets
4. **Performance Issues**: Check debug logs for bottlenecks

### Debug Mode

Enable debug mode for detailed logging:
```matlab
% Set debug mode in main script
DEBUG_MODE = true;
```

### Log Analysis

Use debug reporter to analyze execution:
```matlab
debug_reporter = DebugReporter();
debug_reporter.displayFilteredLog('ERROR', 'VARIABLE');
```

## Contributing

### Code Style
- Follow MATLAB coding conventions
- Use descriptive variable and function names
- Add comprehensive documentation
- Include error handling

### Testing
- Test with various data formats
- Validate error handling
- Check performance with large datasets
- Verify backward compatibility

### Documentation
- Update documentation for new features
- Include usage examples
- Document configuration options
- Maintain troubleshooting guide

## License

This code is part of the sodium channel dynamics analysis project. Please refer to the main project license for usage terms.

## Contact

For questions or issues with the refactored code, please contact the development team or refer to the main project documentation. 