"""
MONET Plots Data Compatibility Test Specifications
=================================================

Comprehensive test specifications for data compatibility requirements
across all MONET Plots classes. Defines xarray/pandas DataFrame compatibility,
data format requirements, and cross-library integration standards with TDD anchors.
All specifications are designed to enable modular, testable fixes under 500 lines each.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
import xarray as xr


class DataCompatibilitySpecifications:
    """
    Data compatibility test specifications for MONET Plots.
    
    This class provides detailed specifications for data format compatibility
    across xarray, pandas DataFrame, and numpy array requirements.
    """
    
    def __init__(self):
        """Initialize data compatibility specifications."""
        self.compatibility_categories = {
            'xarray_compatibility': self._specify_xarray_compatibility(),
            'pandas_compatibility': self._specify_pandas_compatibility(),
            'numpy_compatibility': self._specify_numpy_compatibility(),
            'cross_library_integration': self._specify_cross_library_integration(),
            'data_format_standards': self._specify_data_format_standards()
        }
    
    def _specify_xarray_compatibility(self) -> Dict[str, Any]:
        """
        Specify xarray DataArray and Dataset compatibility requirements.
        
        Defines compatibility standards for xarray integration across plot classes.
        """
        return {
            'dataarray_requirements': {
                'basic_structure': {
                    'required_dimensions': {
                        'minimum_dimensions': 2,
                        'maximum_dimensions': 4,
                        'recommended_dimensions': [2, 3],
                        'dimension_types': ['spatial', 'temporal', 'ensemble', 'variable']
                    },
                    'coordinate_requirements': {
                        'spatial_coords': ['lat', 'lon', 'latitude', 'longitude'],
                        'temporal_coords': ['time', 'date', 'datetime'],
                        'ensemble_coords': ['member', 'ensemble', 'realization'],
                        'required_coords': ['spatial_coords']
                    },
                    'data_structure_validation': [
                        'Must have proper coordinate system',
                        'Dimensions must be named (no unnamed dimensions)',
                        'Coordinates must be monotonic and properly ordered',
                        'Data must have finite values or proper NaN handling'
                    ]
                },
                'metadata_requirements': {
                    'required_attributes': [],
                    'recommended_attributes': ['units', 'long_name', 'standard_name'],
                    'coordinate_attributes': {
                        'lat': ['units: degrees_north', 'standard_name: latitude'],
                        'lon': ['units: degrees_east', 'standard_name: longitude'],
                        'time': ['units: days since', 'calendar: standard']
                    }
                }
            },
            'dataset_requirements': {
                'structure_requirements': {
                    'minimum_variables': 1,
                    'variable_types': ['DataArray', 'coordinate arrays'],
                    'dimension_consistency': 'All variables must share common dimensions'
                },
                'faceting_compatibility': {
                    'required_for_faceting': True,
                    'valid_facet_dimensions': ['time', 'ensemble', 'spatial_subset'],
                    'dimension_validation': [
                        'Facet dimension must exist in dataset',
                        'Facet dimension must have length > 1',
                        'Facet values must be unique'
                    ]
                }
            },
            'xarray_integration_tests': [
                'Valid DataArray with proper coordinates',
                'DataArray with missing coordinates (should fail gracefully)',
                'DataArray with malformed coordinate values',
                'Dataset with multiple variables for faceting',
                'Dataset with missing coordinate system',
                'DataArray with temporal coordinates for time series plots',
                'DataArray with spatial coordinates for spatial plots',
                'Cross-dimensional coordinate relationships'
            ]
        }
    
    def _specify_pandas_compatibility(self) -> Dict[str, Any]:
        """
        Specify pandas DataFrame compatibility requirements.
        
        Defines compatibility standards for pandas DataFrame integration.
        """
        return {
            'dataframe_requirements': {
                'column_structure': {
                    'required_columns': {
                        'timeseries_plots': ['time', 'obs'],
                        'scatter_plots': ['x', 'y'],
                        'statistical_plots': ['value'],
                        'grouping_columns': ['group', 'category', 'label']
                    },
                    'column_validation': [
                        'Required columns must exist',
                        'Column names must be strings',
                        'No duplicate column names',
                        'Columns must contain compatible data types'
                    ],
                    'data_type_requirements': {
                        'numeric_columns': ['int', 'float', 'complex'],
                        'temporal_columns': ['datetime64', 'Timestamp'],
                        'categorical_columns': ['object', 'category'],
                        'forbidden_types': ['mixed', 'unknown']
                    }
                },
                'index_requirements': {
                    'index_types': ['RangeIndex', 'Int64Index', 'DatetimeIndex'],
                    'index_validation': [
                        'Index must be monotonic for time series',
                        'No duplicate index values',
                        'Index values must be finite (no NaN/Inf)'
                    ]
                }
            },
            'data_quality_requirements': {
                'missing_data_handling': {
                    'nan_tolerance': {
                        'timeseries': 'Up to 20% NaN values acceptable',
                        'scatter_plots': 'Up to 10% NaN values acceptable',
                        'statistical_plots': 'Up to 5% NaN values acceptable'
                    },
                    'missing_data_strategies': [
                        'Automatic NaN removal for plotting',
                        'Warning for high missing data percentage',
                        'Option to specify NaN handling strategy'
                    ]
                },
                'data_consistency': {
                    'row_consistency': 'All rows must have same number of columns',
                    'type_consistency': 'Column types must be consistent within column',
                    'index_consistency': 'Index must align with data rows'
                }
            },
            'pandas_integration_tests': [
                'DataFrame with required columns for each plot type',
                'DataFrame with missing required columns (should fail)',
                'DataFrame with incorrect data types',
                'DataFrame with high percentage of NaN values',
                'DataFrame with duplicate rows',
                'DataFrame with non-monotonic time index',
                'DataFrame with categorical data for grouping',
                'DataFrame with multi-level index'
            ]
        }
    
    def _specify_numpy_compatibility(self) -> Dict[str, Any]:
        """
        Specify numpy array compatibility requirements.
        
        Defines compatibility standards for numpy array data input.
        """
        return {
            'array_requirements': {
                'dimensional_requirements': {
                    'spatial_plots': {
                        'minimum_dimensions': 2,
                        'maximum_dimensions': 3,
                        'shape_requirements': ['(height, width)', '(time, height, width)'],
                        'dimension_ordering': 'Row-major (C-style) ordering expected'
                    },
                    'statistical_plots': {
                        'minimum_dimensions': 1,
                        'maximum_dimensions': 2,
                        'shape_requirements': ['(n_samples,)', '(n_samples, n_features)'],
                        'flattening_behavior': 'Multi-dimensional arrays flattened to 1D'
                    },
                    'time_series_plots': {
                        'minimum_dimensions': 1,
                        'maximum_dimensions': 2,
                        'shape_requirements': ['(n_time_points,)', '(n_time_points, n_variables)'],
                        'time_axis_requirement': 'Time axis must be first dimension'
                    }
                },
                'data_type_requirements': {
                    'numeric_types': ['float32', 'float64', 'int32', 'int64'],
                    'forbidden_types': ['object', 'string', 'complex'],
                    'type_conversion': [
                        'Automatic conversion to float64 for integer types',
                        'Error for non-numeric types',
                        'Warning for precision loss in type conversion'
                    ]
                },
                'value_requirements': {
                    'finite_values': {
                        'allowed_values': ['finite numbers', 'NaN', 'Inf'],
                        'value_handling': [
                            'NaN values handled gracefully with warning',
                            'Inf values converted to large finite values',
                            'All-NaN arrays raise informative error'
                        ]
                    },
                    'range_requirements': {
                        'recommended_range': 'Values should be within reasonable physical bounds',
                        'extreme_values': 'Warning for values outside typical ranges',
                        'data_scaling': 'Automatic scaling for extreme value ranges'
                    }
                }
            },
            'numpy_integration_tests': [
                '2D array for spatial plotting',
                '3D array for time-series spatial data',
                '1D array for statistical plots',
                'Array with mixed data types (should fail)',
                'Array with all NaN values (should fail gracefully)',
                'Array with extreme value ranges',
                'Array with different memory layouts',
                'Array with non-contiguous memory'
            ]
        }
    
    def _specify_cross_library_integration(self) -> Dict[str, Any]:
        """
        Specify cross-library data integration requirements.
        
        Defines compatibility standards for data exchange between libraries.
        """
        return {
            'interoperability_standards': {
                'xarray_to_pandas': {
                    'conversion_requirements': {
                        'coordinate_preservation': 'xarray coordinates become DataFrame columns/index',
                        'dimension_preservation': 'Multi-dimensional data properly reshaped',
                        'metadata_preservation': 'Attributes preserved as DataFrame metadata'
                    },
                    'conversion_methods': [
                        'DataArray.to_series() for 1D data',
                        'DataArray.to_dataframe() for multi-dimensional data',
                        'Dataset.to_dataframe() for multi-variable data'
                    ]
                },
                'pandas_to_numpy': {
                    'conversion_requirements': {
                        'column_selection': 'Proper column selection for plot requirements',
                        'index_handling': 'Index converted to appropriate coordinates',
                        'data_extraction': 'Values extracted as numpy arrays'
                    },
                    'conversion_methods': [
                        'DataFrame.values for raw data extraction',
                        'DataFrame.to_numpy() for explicit conversion',
                        'Series.values for single column extraction'
                    ]
                },
                'numpy_to_xarray': {
                    'conversion_requirements': {
                        'coordinate_assignment': 'Proper coordinates assigned to dimensions',
                        'dimension_naming': 'Dimensions given meaningful names',
                        'metadata_addition': 'Appropriate attributes added'
                    },
                    'conversion_methods': [
                        'xr.DataArray() constructor',
                        'xr.Dataset() for multiple variables',
                        'Coordinate specification via coords parameter'
                    ]
                }
            },
            'data_exchange_tests': [
                'xarray DataArray to pandas DataFrame conversion',
                'pandas DataFrame to numpy array extraction',
                'numpy array to xarray DataArray creation',
                'Round-trip conversion xarray -> pandas -> numpy -> xarray',
                'Data integrity verification after conversions',
                'Metadata preservation during conversions',
                'Coordinate system preservation',
                'Missing data handling consistency across libraries'
            ]
        }
    
    def _specify_data_format_standards(self) -> Dict[str, Any]:
        """
        Specify general data format standards and validation.
        
        Defines overarching data format requirements and validation strategies.
        """
        return {
            'format_validation': {
                'automatic_detection': {
                    'data_type_detection': [
                        'Automatic detection of xarray, pandas, numpy data types',
                        'Intelligent format inference from data structure',
                        'Fallback detection for ambiguous data types'
                    ],
                    'format_validation': [
                        'Validation of data structure integrity',
                        'Verification of required metadata',
                        'Consistency checks for multi-dimensional data'
                    ]
                },
                'validation_strategies': {
                    'structural_validation': [
                        'Check data dimensions and shape',
                        'Verify coordinate systems and indices',
                        'Validate data type consistency'
                    ],
                    'content_validation': [
                        'Check for data completeness',
                        'Validate value ranges and types',
                        'Verify metadata completeness'
                    ],
                    'compatibility_validation': [
                        'Test data compatibility with plot requirements',
                        'Validate cross-library data exchange',
                        'Check for potential conversion issues'
                    ]
                }
            },
            'error_handling_strategies': {
                'format_errors': {
                    'invalid_format': {
                        'detection': 'Clear identification of invalid data formats',
                        'error_message': 'Informative error messages with format requirements',
                        'suggestions': 'Specific suggestions for format correction'
                    },
                    'missing_requirements': {
                        'detection': 'Identification of missing required components',
                        'error_message': 'Clear indication of missing elements',
                        'examples': 'Examples of correct format'
                    }
                },
                'conversion_errors': {
                    'type_conversion': {
                        'precision_loss': 'Warning for potential precision loss',
                        'data_loss': 'Error for significant data loss during conversion',
                        'type_incompatibility': 'Error for incompatible type conversions'
                    },
                    'structure_conversion': {
                        'dimension_mismatch': 'Error for incompatible dimension structures',
                        'coordinate_mismatch': 'Error for coordinate system conflicts',
                        'metadata_loss': 'Warning for metadata loss during conversion'
                    }
                }
            },
            'data_format_tests': [
                'Automatic data format detection',
                'Validation of mixed data type inputs',
                'Error handling for invalid data formats',
                'Conversion between different data formats',
                'Data integrity verification after format operations',
                'Performance testing for large data format conversions',
                'Memory usage optimization during data format operations',
                'Edge case handling for unusual data structures'
            ]
        }


# TDD Anchor Point: Data Compatibility Specification Definition
# PURPOSE: Define comprehensive data compatibility requirements across all libraries
# VALIDATION: All specifications must meet modular design principles
# REQUIREMENT: Each specification module < 500 lines

def get_data_compatibility_specifications() -> Dict[str, Any]:
    """
    Get all data compatibility specifications.
    
    Returns:
        Dictionary containing all data compatibility specifications.
    """
    specs = DataCompatibilitySpecifications()
    return specs.compatibility_categories


def validate_data_format(data: Any, expected_format: str) -> Dict[str, Any]:
    """
    Validate data format against expected format specifications.
    
    Args:
        data: Data object to validate
        expected_format: Expected format ('xarray', 'pandas', 'numpy', 'auto')
    
    Returns:
        Dictionary containing validation results and any issues found.
    """
    # This would contain actual validation logic based on the specifications
    # For specification purposes, returning a placeholder structure
    return {
        'data_type': type(data).__name__,
        'expected_format': expected_format,
        'validation_passed': True,
        'issues_found': [],
        'suggestions': [],
        'compatibility_score': 1.0
    }


def get_compatibility_requirements(library: str, plot_type: str) -> Dict[str, Any]:
    """
    Get compatibility requirements for a specific library and plot type.
    
    Args:
        library: Data library ('xarray', 'pandas', 'numpy')
        plot_type: Type of plot being created
    
    Returns:
        Dictionary containing compatibility requirements.
    """
    specs = DataCompatibilitySpecifications()
    library_specs = specs.compatibility_categories.get(f'{library}_compatibility', {})
    
    # Extract requirements relevant to the plot type
    requirements = {}
    if 'dataframe_requirements' in library_specs:
        requirements = library_specs['dataframe_requirements']
    elif 'dataarray_requirements' in library_specs:
        requirements = library_specs['dataarray_requirements']
    elif 'array_requirements' in library_specs:
        requirements = library_specs['array_requirements']
    
    return requirements


def generate_compatibility_test_cases() -> List[Dict[str, Any]]:
    """
    Generate comprehensive test cases for data compatibility validation.
    
    Returns:
        List of test case dictionaries with data format and validation requirements.
    """
    specs = DataCompatibilitySpecifications()
    
    test_cases = []
    
    # Generate test cases from each compatibility category
    for category_name, category_specs in specs.compatibility_categories.items():
        if 'tests' in category_specs:
            for test_description in category_specs['tests']:
                test_cases.append({
                    'category': category_name,
                    'test_description': test_description,
                    'data_format': category_name.replace('_compatibility', ''),
                    'expected_outcome': 'success' if 'valid' in test_description.lower() else 'failure',
                    'validation_strategy': 'format_validation'
                })
        elif isinstance(category_specs, dict):
            for subcategory_name, subcategory_specs in category_specs.items():
                if isinstance(subcategory_specs, dict) and 'tests' in subcategory_specs:
                    for test_description in subcategory_specs['tests']:
                        test_cases.append({
                            'category': f"{category_name}.{subcategory_name}",
                            'test_description': test_description,
                            'data_format': category_name.replace('_compatibility', ''),
                            'expected_outcome': 'success' if 'valid' in test_description.lower() else 'failure',
                            'validation_strategy': 'format_validation'
                        })
    
    return test_cases


def get_data_conversion_guidelines() -> Dict[str, Any]:
    """
    Get guidelines for data conversion between different formats.
    
    Returns:
        Dictionary containing conversion guidelines and best practices.
    """
    specs = DataCompatibilitySpecifications()
    return specs.compatibility_categories.get('cross_library_integration', {})


def validate_cross_library_compatibility(data: Any, target_library: str) -> Dict[str, Any]:
    """
    Validate data compatibility for conversion to a target library.
    
    Args:
        data: Source data object
        target_library: Target library ('xarray', 'pandas', 'numpy')
    
    Returns:
        Dictionary containing compatibility validation results.
    """
    # This would contain actual cross-library validation logic
    return {
        'source_type': type(data).__name__,
        'target_library': target_library,
        'conversion_possible': True,
        'data_loss_risk': 'low',
        'metadata_preservation': 'complete',
        'performance_impact': 'minimal'
    }


# Example usage and validation
if __name__ == "__main__":
    # Print summary of all data compatibility specifications
    compatibility_specs = get_data_compatibility_specifications()
    
    print("MONET Plots Data Compatibility Specifications Summary")
    print("=" * 55)
    
    for category, specs in compatibility_specs.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        if isinstance(specs, dict) and 'tests' in specs:
            print(f"  Test scenarios: {len(specs['tests'])} defined")
        elif isinstance(specs, dict):
            print(f"  Subcategories: {len(specs)} defined")
            
            # Count total test scenarios
            total_tests = 0
            for subcategory, sub_specs in specs.items():
                if isinstance(sub_specs, dict) and 'tests' in sub_specs:
                    total_tests += len(sub_specs['tests'])
            if total_tests > 0:
                print(f"  Test scenarios: {total_tests}")
    
    print(f"\nTotal compatibility categories: {len(compatibility_specs)}")
    print("Data compatibility specifications ready for TDD implementation.")
    
    # Example validation
    example_validation = validate_data_format(pd.DataFrame({'x': [1, 2], 'y': [3, 4]}), 'pandas')
    print(f"Example validation result: {example_validation['validation_passed']}")