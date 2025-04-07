#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Environment diagnostic tool for LLM Finetuning Framework.
This script performs checks on the system environment, installed packages,
GPU availability, and workspace to identify potential issues.
"""

import os
import sys
import json
import platform
import subprocess
import importlib
import logging
from pathlib import Path
from datetime import datetime
import tempfile

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)

logger = logging.getLogger("diagnose-environment")

# Define required packages with their minimum versions
REQUIRED_PACKAGES = {
    "numpy": "1.20.0",
    "torch": "2.0.0",
    "transformers": "4.30.0",
    "datasets": "2.12.0",
    "peft": "0.4.0",
    "evaluate": "0.4.0",
    "scikit-learn": "1.0.0",
}

# Optional packages that enhance functionality
OPTIONAL_PACKAGES = {
    "accelerate": "0.20.0",
    "bitsandbytes": "0.39.0",
    "tensorboard": "2.12.0",
    "wandb": "0.15.0",
    "hyperopt": "0.2.7",
    "ray": "2.4.0",
}

def run_command(cmd):
    """Run a shell command and return the output."""
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=False, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            text=True
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), -1

def check_package(package_name, min_version=None):
    """Check if a package is installed and meets the minimum version requirement."""
    try:
        # Try to import the package
        module = importlib.import_module(package_name)
        
        if hasattr(module, "__version__"):
            version = module.__version__
        elif hasattr(module, "VERSION"):
            version = module.VERSION
        elif hasattr(module, "version"):
            version = module.version
        else:
            # Try to get version using pkg_resources
            try:
                import pkg_resources
                version = pkg_resources.get_distribution(package_name).version
            except:
                version = "Unknown"
                
        # Check version if minimum is specified
        if min_version and version != "Unknown":
            from packaging import version as packaging_version
            if packaging_version.parse(version) < packaging_version.parse(min_version):
                return False, version, f"Version {version} is lower than required {min_version}"
        
        return True, version, "OK"
    except ImportError:
        return False, None, "Not installed"
    except Exception as e:
        return False, None, str(e)

def check_gpu():
    """Check if GPU is available with CUDA."""
    gpu_info = {
        "torch_cuda_available": False,
        "cuda_version": None,
        "gpu_count": 0,
        "gpu_names": [],
        "gpu_memory": []
    }
    
    # Check CUDA availability in PyTorch
    try:
        import torch
        gpu_info["torch_cuda_available"] = torch.cuda.is_available()
        if gpu_info["torch_cuda_available"]:
            gpu_info["cuda_version"] = torch.version.cuda
            gpu_info["gpu_count"] = torch.cuda.device_count()
            
            for i in range(gpu_info["gpu_count"]):
                gpu_info["gpu_names"].append(torch.cuda.get_device_name(i))
                # Get GPU memory
                try:
                    total_mem = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # Convert to GB
                    gpu_info["gpu_memory"].append(f"{total_mem:.2f} GB")
                except:
                    gpu_info["gpu_memory"].append("Unknown")
    except:
        pass
    
    return gpu_info

def check_disk_space(path="."):
    """Check available disk space."""
    try:
        if platform.system() == "Windows":
            import ctypes
            free_bytes = ctypes.c_ulonglong(0)
            total_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(path), None, ctypes.pointer(total_bytes), ctypes.pointer(free_bytes)
            )
            total_gb = total_bytes.value / (1024**3)
            free_gb = free_bytes.value / (1024**3)
        else:
            import shutil
            total, used, free = shutil.disk_usage(path)
            total_gb = total / (1024**3)
            free_gb = free / (1024**3)
            
        return {
            "total_gb": round(total_gb, 2),
            "free_gb": round(free_gb, 2),
            "free_percent": round((free_gb / total_gb) * 100, 2)
        }
    except Exception as e:
        return {"error": str(e)}

def check_configs():
    """Check if configuration files exist and are valid JSON."""
    config_files = [
        "validation/quick_test_config.json",
        "validation/synthetic_test_config.json"
    ]
    
    results = {}
    for config_file in config_files:
        try:
            if not os.path.exists(config_file):
                results[config_file] = {"exists": False, "valid": False, "error": "File not found"}
                continue
                
            with open(config_file, 'r') as f:
                json.load(f)
            results[config_file] = {"exists": True, "valid": True}
        except json.JSONDecodeError as e:
            results[config_file] = {"exists": True, "valid": False, "error": f"Invalid JSON: {str(e)}"}
        except Exception as e:
            results[config_file] = {"exists": True, "valid": False, "error": str(e)}
    
    return results

def check_write_permissions():
    """Check if we have write permissions in the necessary directories."""
    dirs_to_check = [
        ".",  # Current directory
        "validation",
        "data",
        "validation/outputs"
    ]
    
    results = {}
    for dir_path in dirs_to_check:
        # Create dir if it doesn't exist
        if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path, exist_ok=True)
            except Exception as e:
                results[dir_path] = {"can_write": False, "error": f"Cannot create directory: {str(e)}"}
                continue
        
        # Check if we can write to the directory
        try:
            test_file = os.path.join(dir_path, f".test_write_{datetime.now().strftime('%Y%m%d%H%M%S')}")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            results[dir_path] = {"can_write": True}
        except Exception as e:
            results[dir_path] = {"can_write": False, "error": str(e)}
    
    return results

def check_workspace_structure():
    """Check if the workspace has the expected directory structure."""
    expected_dirs = [
        "data",
        "models",
        "pipeline",
        "validation"
    ]
    
    expected_files = [
        "requirements.txt",
        "README.md"
    ]
    
    results = {
        "directories": {},
        "files": {}
    }
    
    for directory in expected_dirs:
        exists = os.path.isdir(directory)
        results["directories"][directory] = {"exists": exists}
    
    for file in expected_files:
        exists = os.path.isfile(file)
        results["files"][file] = {"exists": exists}
    
    return results

def check_data_contents():
    """Check if necessary data directories and files exist."""
    synthetic_data_path = "data/synthetic"
    
    results = {
        "synthetic_data": {
            "exists": os.path.isdir(synthetic_data_path),
            "splits": {}
        }
    }
    
    if results["synthetic_data"]["exists"]:
        for split in ["train", "validation", "test"]:
            split_path = os.path.join(synthetic_data_path, split)
            results["synthetic_data"]["splits"][split] = {
                "exists": os.path.isdir(split_path)
            }
            
            if results["synthetic_data"]["splits"][split]["exists"]:
                try:
                    files = os.listdir(split_path)
                    results["synthetic_data"]["splits"][split]["files"] = files
                except Exception as e:
                    results["synthetic_data"]["splits"][split]["error"] = str(e)
    
    return results

def check_huggingface_cache():
    """Check Hugging Face cache location and size."""
    try:
        from transformers.utils import TRANSFORMERS_CACHE
        cache_dir = TRANSFORMERS_CACHE
    except:
        # Default cache locations
        if platform.system() == "Windows":
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
        else:
            cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "huggingface")
    
    results = {
        "cache_dir": cache_dir,
        "exists": os.path.isdir(cache_dir),
        "size_gb": None,
        "can_write": None
    }
    
    if results["exists"]:
        # Calculate cache size
        try:
            total_size = 0
            for dirpath, _, filenames in os.walk(cache_dir):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if not os.path.islink(fp):
                        total_size += os.path.getsize(fp)
            
            results["size_gb"] = round(total_size / (1024**3), 2)
        except Exception as e:
            results["size_error"] = str(e)
        
        # Check write permission
        try:
            test_file = os.path.join(cache_dir, f".test_write_{datetime.now().strftime('%Y%m%d%H%M%S')}")
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            results["can_write"] = True
        except Exception as e:
            results["can_write"] = False
            results["write_error"] = str(e)
    
    return results

def check_memory():
    """Check available system memory."""
    try:
        import psutil
        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / (1024**3), 2),
            "available_gb": round(mem.available / (1024**3), 2),
            "percent_used": mem.percent
        }
    except ImportError:
        # Fallback if psutil is not available
        if platform.system() == "Linux":
            meminfo = {}
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    key, value = line.split(':', 1)
                    meminfo[key.strip()] = value.strip()
            
            total_kb = int(meminfo['MemTotal'].split()[0])
            available_kb = int(meminfo.get('MemAvailable', meminfo['MemFree']).split()[0])
            
            return {
                "total_gb": round(total_kb / (1024**2), 2),
                "available_gb": round(available_kb / (1024**2), 2),
                "percent_used": round(((total_kb - available_kb) / total_kb) * 100, 2)
            }
        elif platform.system() == "Windows":
            # Try using wmic on Windows
            stdout, stderr, returncode = run_command("wmic OS get TotalVisibleMemorySize,FreePhysicalMemory /Value")
            if returncode == 0 and stdout:
                lines = stdout.split('\n')
                total_kb = int([line.split('=')[1] for line in lines if "TotalVisibleMemorySize" in line][0])
                free_kb = int([line.split('=')[1] for line in lines if "FreePhysicalMemory" in line][0])
                
                return {
                    "total_gb": round(total_kb / (1024**2), 2),
                    "available_gb": round(free_kb / (1024**2), 2),
                    "percent_used": round(((total_kb - free_kb) / total_kb) * 100, 2)
                }
            
    return {"error": "Could not determine memory info. Install psutil for better results."}

def check_pytorch_cpu():
    """Verify PyTorch CPU functionality by running a simple operation."""
    try:
        import torch
        
        # Create a simple tensor
        x = torch.tensor([1.0, 2.0, 3.0])
        y = torch.tensor([4.0, 5.0, 6.0])
        
        # Perform operations
        z = x + y
        dot_product = torch.dot(x, y)
        
        return {
            "functional": True,
            "result": dot_product.item()
        }
    except Exception as e:
        return {
            "functional": False,
            "error": str(e)
        }

def check_pytorch_gpu():
    """Verify PyTorch GPU functionality by running a simple operation."""
    try:
        import torch
        
        if not torch.cuda.is_available():
            return {
                "functional": False,
                "error": "CUDA not available"
            }
        
        # Create a simple tensor on GPU
        x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
        y = torch.tensor([4.0, 5.0, 6.0], device="cuda")
        
        # Perform operations
        z = x + y
        dot_product = torch.dot(x, y)
        
        return {
            "functional": True,
            "result": dot_product.item()
        }
    except Exception as e:
        return {
            "functional": False,
            "error": str(e)
        }

def main():
    """Run all diagnostics and print results."""
    logger.info("Starting environment diagnostics...")
    
    # Create diagnostics report
    report = {
        "timestamp": datetime.now().isoformat(),
        "system_info": {
            "os": platform.system(),
            "os_version": platform.version(),
            "python_version": platform.python_version(),
            "python_path": sys.executable,
        },
        "memory": check_memory(),
        "disk_space": check_disk_space(),
        "gpu_info": check_gpu(),
        "required_packages": {},
        "optional_packages": {},
        "pytorch_cpu_test": check_pytorch_cpu(),
        "pytorch_gpu_test": check_pytorch_gpu() if check_gpu()["torch_cuda_available"] else {"functional": False, "error": "CUDA not available"},
        "workspace_structure": check_workspace_structure(),
        "write_permissions": check_write_permissions(),
        "configs": check_configs(),
        "data_contents": check_data_contents(),
        "huggingface_cache": check_huggingface_cache()
    }
    
    # Check required packages
    for package, min_version in REQUIRED_PACKAGES.items():
        installed, version, message = check_package(package, min_version)
        report["required_packages"][package] = {
            "installed": installed,
            "version": version,
            "min_version": min_version,
            "message": message
        }
    
    # Check optional packages
    for package, min_version in OPTIONAL_PACKAGES.items():
        installed, version, message = check_package(package, min_version)
        report["optional_packages"][package] = {
            "installed": installed,
            "version": version,
            "min_version": min_version,
            "message": message
        }
    
    # Print summary to console
    logger.info("=== LLM Finetuning Environment Diagnostics ===")
    logger.info(f"System: {report['system_info']['os']} {report['system_info']['os_version']}")
    logger.info(f"Python: {report['system_info']['python_version']} ({report['system_info']['python_path']})")
    
    # Memory and disk
    logger.info(f"\n=== System Resources ===")
    logger.info(f"Memory: {report['memory'].get('available_gb', 'Unknown')} GB available of {report['memory'].get('total_gb', 'Unknown')} GB total")
    logger.info(f"Disk Space: {report['disk_space'].get('free_gb', 'Unknown')} GB free of {report['disk_space'].get('total_gb', 'Unknown')} GB total")
    
    # GPU
    logger.info(f"\n=== GPU Information ===")
    if report['gpu_info']['torch_cuda_available']:
        logger.info(f"CUDA available: Yes (version {report['gpu_info']['cuda_version']})")
        logger.info(f"GPU count: {report['gpu_info']['gpu_count']}")
        for i, (name, memory) in enumerate(zip(report['gpu_info']['gpu_names'], report['gpu_info']['gpu_memory'])):
            logger.info(f"  GPU {i}: {name} ({memory})")
        
        if report['pytorch_gpu_test']['functional']:
            logger.info("GPU functionality: OK")
        else:
            logger.info(f"GPU functionality: FAILED - {report['pytorch_gpu_test'].get('error', 'Unknown error')}")
    else:
        logger.info("CUDA available: No")
    
    # Required packages
    logger.info(f"\n=== Required Packages ===")
    missing_required = []
    for package, info in report["required_packages"].items():
        status = "✓" if info["installed"] else "✗"
        version_info = f"v{info['version']}" if info["version"] else "not installed"
        logger.info(f"{status} {package}: {version_info}")
        if not info["installed"]:
            missing_required.append(package)
    
    # Optional packages
    logger.info(f"\n=== Optional Packages ===")
    for package, info in report["optional_packages"].items():
        status = "✓" if info["installed"] else "-"
        version_info = f"v{info['version']}" if info["version"] else "not installed"
        logger.info(f"{status} {package}: {version_info}")
    
    # Workspace structure
    logger.info(f"\n=== Workspace Structure ===")
    for dir_name, info in report["workspace_structure"]["directories"].items():
        status = "✓" if info["exists"] else "✗"
        logger.info(f"{status} Directory: {dir_name}")
    
    for file_name, info in report["workspace_structure"]["files"].items():
        status = "✓" if info["exists"] else "✗"
        logger.info(f"{status} File: {file_name}")
    
    # Configuration files
    logger.info(f"\n=== Configuration Files ===")
    for config_file, info in report["configs"].items():
        if info["exists"] and info["valid"]:
            logger.info(f"✓ {config_file}: Valid JSON")
        elif info["exists"] and not info["valid"]:
            logger.info(f"✗ {config_file}: Invalid JSON - {info.get('error', 'Unknown error')}")
        else:
            logger.info(f"✗ {config_file}: File not found")
    
    # Write permissions
    logger.info(f"\n=== Write Permissions ===")
    for dir_path, info in report["write_permissions"].items():
        status = "✓" if info["can_write"] else "✗"
        message = "" if info["can_write"] else f" - {info.get('error', 'Permission denied')}"
        logger.info(f"{status} {dir_path}{message}")
    
    # Data contents
    logger.info(f"\n=== Synthetic Data ===")
    if report["data_contents"]["synthetic_data"]["exists"]:
        logger.info("✓ Synthetic data directory exists")
        for split, info in report["data_contents"]["synthetic_data"]["splits"].items():
            status = "✓" if info["exists"] else "✗"
            logger.info(f"  {status} {split} split")
    else:
        logger.info("✗ Synthetic data directory not found")
        logger.info("  Run 'python validation/generate_synthetic_data.py' to create the dataset")
    
    # HuggingFace cache
    logger.info(f"\n=== Hugging Face Cache ===")
    if report["huggingface_cache"]["exists"]:
        logger.info(f"✓ Cache directory: {report['huggingface_cache']['cache_dir']}")
        if report["huggingface_cache"]["size_gb"] is not None:
            logger.info(f"  Size: {report['huggingface_cache']['size_gb']} GB")
        if report["huggingface_cache"]["can_write"] is not None:
            status = "✓" if report["huggingface_cache"]["can_write"] else "✗"
            logger.info(f"  {status} Write permission")
    else:
        logger.info(f"✗ Cache directory not found: {report['huggingface_cache']['cache_dir']}")
    
    # Overall assessment
    logger.info(f"\n=== Overall Assessment ===")
    
    # Critical issues
    critical_issues = []
    if missing_required:
        critical_issues.append(f"Missing required packages: {', '.join(missing_required)}")
    
    if not report["pytorch_cpu_test"]["functional"]:
        critical_issues.append(f"PyTorch CPU not functional: {report['pytorch_cpu_test'].get('error', 'Unknown error')}")
    
    if report["gpu_info"]["torch_cuda_available"] and not report["pytorch_gpu_test"]["functional"]:
        critical_issues.append(f"PyTorch GPU not functional: {report['pytorch_gpu_test'].get('error', 'Unknown error')}")
    
    for dir_path, info in report["write_permissions"].items():
        if not info["can_write"] and dir_path in [".", "validation", "data", "validation/outputs"]:
            critical_issues.append(f"Cannot write to {dir_path}: {info.get('error', 'Permission denied')}")
    
    for config_file, info in report["configs"].items():
        if not info["exists"] or not info["valid"]:
            critical_issues.append(f"Configuration issue: {config_file} - {info.get('error', 'Not found or invalid')}")
    
    # Warnings
    warnings = []
    if report["memory"].get("available_gb", 0) < 4:
        warnings.append(f"Low memory: {report['memory'].get('available_gb', 'Unknown')} GB available (min recommended: 4 GB)")
    
    if report["disk_space"].get("free_gb", 0) < 10:
        warnings.append(f"Low disk space: {report['disk_space'].get('free_gb', 'Unknown')} GB free (min recommended: 10 GB)")
    
    if not report["data_contents"]["synthetic_data"]["exists"]:
        warnings.append("Synthetic data not found. Run 'python validation/generate_synthetic_data.py' to create it.")
    
    # Print assessment
    if critical_issues:
        logger.error("CRITICAL ISSUES FOUND:")
        for issue in critical_issues:
            logger.error(f"  ✗ {issue}")
    
    if warnings:
        logger.warning("WARNINGS:")
        for warning in warnings:
            logger.warning(f"  ! {warning}")
    
    if not critical_issues and not warnings:
        logger.info("✓ No critical issues found. Environment looks good!")
    elif not critical_issues:
        logger.info("✓ No critical issues found, but there are some warnings.")
    
    # Save report to file
    report_file = os.path.join("validation", "diagnostic_report.json")
    try:
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"\nDetailed report saved to: {report_file}")
    except Exception as e:
        logger.error(f"Failed to save report: {str(e)}")
    
    # Provide recommendations based on issues
    if critical_issues or warnings:
        logger.info("\n=== Recommendations ===")
        
        if any("Missing required packages" in issue for issue in critical_issues):
            logger.info("1. Install missing dependencies:")
            logger.info("   python validation/setup_validation.py")
            logger.info("   # Or for all dependencies:")
            logger.info("   pip install -r requirements.txt")
        
        if any("Configuration issue" in issue for issue in critical_issues):
            logger.info("2. Validate your configuration files:")
            logger.info("   python validation/validate_config.py validation/quick_test_config.json")
            logger.info("   python validation/validate_config.py validation/synthetic_test_config.json")
        
        if not report["data_contents"]["synthetic_data"]["exists"]:
            logger.info("3. Generate synthetic data:")
            logger.info("   python validation/generate_synthetic_data.py")
        
        if any("Cannot write" in issue for issue in critical_issues):
            logger.info("4. Fix permission issues:")
            if platform.system() == "Windows":
                logger.info("   # Run as administrator or check folder permissions")
            else:
                logger.info("   chmod -R u+w /path/to/workspace")
        
        if report["gpu_info"]["torch_cuda_available"] and not report["pytorch_gpu_test"]["functional"]:
            logger.info("5. Fix GPU issues:")
            logger.info("   # Check CUDA installation and verify PyTorch was installed with CUDA support")
            logger.info("   pip uninstall torch")
            logger.info("   pip install torch --index-url https://download.pytorch.org/whl/cu118")
            logger.info("   # If using pip, install torch with CUDA support")

if __name__ == "__main__":
    main() 