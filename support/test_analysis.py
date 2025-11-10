import sys
import os
sys.path.insert(0, r'Z:\Shared\Amichai\Code\DB')

# Force output flushing
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

print("Starting analysis script...", flush=True)

try:
    from support.mp4_audio_analysis import analyze_mp4_audio
    print("Module imported successfully", flush=True)
except Exception as e:
    print(f"Import error: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Analyze the provided M4A file
mp4_file = r"C:\Users\Owner\Downloads\10 Nov at 16-20.m4a"

print(f"File path: {mp4_file}", flush=True)
print(f"File exists: {os.path.exists(mp4_file)}", flush=True)

try:
    print("Starting analysis...", flush=True)
    results = analyze_mp4_audio(mp4_file, plot=True)
    print("\nAnalysis complete!", flush=True)
    print("\nDisplaying plots...", flush=True)
    results['amplitude_plot'].show()
    results['spectrum_plot'].show()
    print("Plots should be displayed in your browser.", flush=True)
except Exception as e:
    print(f"Error: {e}", flush=True)
    import traceback
    traceback.print_exc()
    sys.exit(1)

