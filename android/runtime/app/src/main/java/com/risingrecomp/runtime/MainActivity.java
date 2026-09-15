package com.risingrecomp.runtime;

import android.app.Activity;
import android.graphics.Color;
import android.os.Bundle;
import android.view.Gravity;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;

public final class MainActivity extends Activity implements SurfaceHolder.Callback {
    static {
        System.loadLibrary("rising_runtime_bootstrap");
    }

    private TextView report;
    private SurfaceView surface;

    private static native String nativeStart(Object surface);
    private static native String nativeStop();

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        buildInterface();
    }

    private int dp(int value) {
        return Math.round(value * getResources().getDisplayMetrics().density);
    }

    private TextView text(String value, float size, int color) {
        TextView view = new TextView(this);
        view.setText(value);
        view.setTextSize(size);
        view.setTextColor(color);
        return view;
    }

    private void buildInterface() {
        LinearLayout page = new LinearLayout(this);
        page.setOrientation(LinearLayout.VERTICAL);
        page.setPadding(dp(18), dp(18), dp(18), dp(18));
        page.setBackgroundColor(Color.rgb(250, 250, 252));

        TextView title = text("RisingRecomp", 27, Color.rgb(25, 25, 28));
        page.addView(title);
        TextView subtitle = text("Android native runtime bootstrap — game not loaded", 14,
                Color.rgb(82, 82, 90));
        subtitle.setPadding(0, dp(2), 0, dp(10));
        page.addView(subtitle);

        surface = new SurfaceView(this);
        surface.setBackgroundColor(Color.rgb(16, 17, 20));
        surface.getHolder().addCallback(this);
        page.addView(surface, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(132)));

        report = text("Waiting for Android surface…", 13, Color.rgb(35, 35, 40));
        report.setTypeface(android.graphics.Typeface.MONOSPACE);
        report.setPadding(dp(10), dp(10), dp(10), dp(10));
        report.setTextIsSelectable(true);
        ScrollView scroll = new ScrollView(this);
        scroll.addView(report);
        page.addView(scroll, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f));

        Button restart = new Button(this);
        restart.setText("Reiniciar runtime nativo");
        restart.setAllCaps(false);
        restart.setGravity(Gravity.CENTER);
        restart.setOnClickListener(ignored -> report.setText(nativeStart(surface.getHolder().getSurface())));
        page.addView(restart, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT));
        setContentView(page);
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        report.setText(nativeStart(holder.getSurface()));
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        // The later swapchain stage owns resize handling. The bootstrap only records a live window.
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        report.setText(nativeStop());
    }

    @Override
    protected void onDestroy() {
        nativeStop();
        super.onDestroy();
    }
}
