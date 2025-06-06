#!/usr/bin/env python3
"""
Training Progress Monitor

Monitors the latest YOLO training progress by checking log files and results.
"""

import os
from pathlib import Path
import time
import glob

def find_latest_training_dir():
    """Find the most recent training directory."""
    train_dirs = glob.glob("runs/detect/enhanced_train_*")
    if not train_dirs:
        return None
    
    # Sort by modification time, most recent first
    train_dirs.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return Path(train_dirs[0])

def monitor_training():
    """Monitor the training progress."""
    print("🔍 YOLO Training Monitor")
    print("=" * 50)
    
    latest_dir = find_latest_training_dir()
    if not latest_dir:
        print("❌ No training directory found")
        return
    
    print(f"📁 Monitoring: {latest_dir}")
    print(f"🕐 Started at: {time.ctime(os.path.getmtime(latest_dir))}")
    
    # Check for results files
    results_file = latest_dir / "results.csv"
    weights_dir = latest_dir / "weights"
    
    print(f"\n📊 Training Status:")
    
    if results_file.exists():
        print(f"✅ Results file exists: {results_file}")
        
        # Read last few lines of results
        try:
            with open(results_file, 'r') as f:
                lines = f.readlines()
            
            if len(lines) > 1:
                headers = lines[0].strip().split(',')
                latest_data = lines[-1].strip().split(',')
                
                if len(latest_data) >= 4:
                    epoch = latest_data[0].strip()
                    train_loss = latest_data[1].strip() if len(latest_data) > 1 else "N/A"
                    val_loss = latest_data[2].strip() if len(latest_data) > 2 else "N/A"
                    map50 = latest_data[6].strip() if len(latest_data) > 6 else "N/A"
                    
                    print(f"   📈 Current Epoch: {epoch}")
                    print(f"   📉 Train Loss: {train_loss}")
                    print(f"   📉 Val Loss: {val_loss}")
                    print(f"   🎯 mAP50: {map50}")
                else:
                    print(f"   📈 Total epochs logged: {len(lines) - 1}")
        except Exception as e:
            print(f"   ⚠️  Error reading results: {e}")
    else:
        print(f"❌ Results file not found yet")
    
    if weights_dir.exists():
        weights = list(weights_dir.glob("*.pt"))
        print(f"\n💾 Saved Weights: {len(weights)} files")
        for weight in weights:
            print(f"   📦 {weight.name}")
    else:
        print(f"❌ Weights directory not found yet")
    
    # Check for completion
    best_pt = latest_dir / "weights" / "best.pt"
    if best_pt.exists():
        print(f"\n🎉 Training appears complete!")
        print(f"   🏆 Best model: {best_pt}")
        
        # Try to read final results
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    lines = f.readlines()
                
                if len(lines) > 1:
                    final_data = lines[-1].strip().split(',')
                    if len(final_data) >= 7:
                        final_map50 = final_data[6].strip()
                        print(f"   📊 Final mAP50: {final_map50}")
                        print(f"   📈 Baseline was: 0.972")
                        
                        try:
                            improvement = float(final_map50) - 0.972
                            if improvement > 0:
                                print(f"   ✅ Improvement: +{improvement:.3f}")
                            else:
                                print(f"   📊 Change: {improvement:.3f}")
                        except:
                            pass
            except Exception as e:
                print(f"   ⚠️  Error reading final results: {e}")
    else:
        print(f"\n⏳ Training in progress...")
    
    print(f"\n📁 Full training directory: {latest_dir.absolute()}")

if __name__ == "__main__":
    monitor_training() 