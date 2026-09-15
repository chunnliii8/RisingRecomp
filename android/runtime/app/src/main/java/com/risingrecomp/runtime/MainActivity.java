package com.risingrecomp.runtime;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.os.ParcelFileDescriptor;
import android.view.Gravity;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public final class MainActivity extends Activity implements SurfaceHolder.Callback {
    private static final int PICK_XBLA = 2001;
    private static final int EXPORT_XEX = 2002;
    private static final String PREFS = "risingrecomp_runtime";
    private static final String PREF_PACKAGE_URI = "package_uri";

    static {
        System.loadLibrary("rising_runtime_bootstrap");
    }

    private final ExecutorService worker = Executors.newSingleThreadExecutor();
    private TextView report;
    private SurfaceView surface;
    private Button selectPackage;
    private Button exportXex;
    private String runtimeReport = "Waiting for Android surface…\n";
    private String intakeReport = "No XBLA package selected.\n";
    private File privateXex;
    private File privateManifest;

    private static native String nativeStart(Object surface);
    private static native String nativeStop();
    private static native String nativeInspectPackage(int fd, String xexPath,
                                                       String manifestPath);

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        File intakeDirectory = new File(getFilesDir(), "game-intake");
        if (!intakeDirectory.exists() && !intakeDirectory.mkdirs()) {
            Toast.makeText(this, "Falha ao criar armazenamento privado", Toast.LENGTH_LONG).show();
        }
        privateXex = new File(intakeDirectory, "default.xex");
        privateManifest = new File(intakeDirectory, "package-manifest.txt");
        buildInterface();
        exportXex.setEnabled(false);
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

    private Button button(String label) {
        Button button = new Button(this);
        button.setText(label);
        button.setAllCaps(false);
        button.setGravity(Gravity.CENTER);
        return button;
    }

    private void buildInterface() {
        LinearLayout page = new LinearLayout(this);
        page.setOrientation(LinearLayout.VERTICAL);
        page.setPadding(dp(18), dp(14), dp(18), dp(12));
        page.setBackgroundColor(Color.rgb(250, 250, 252));

        page.addView(text("RisingRecomp", 27, Color.rgb(25, 25, 28)));
        TextView subtitle = text("Stage 3 — local Case Zero XBLA intake", 14,
                Color.rgb(82, 82, 90));
        subtitle.setPadding(0, dp(2), 0, dp(8));
        page.addView(subtitle);

        surface = new SurfaceView(this);
        surface.setBackgroundColor(Color.rgb(16, 17, 20));
        surface.getHolder().addCallback(this);
        page.addView(surface, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, dp(96)));

        report = text("Starting…", 12, Color.rgb(35, 35, 40));
        report.setTypeface(android.graphics.Typeface.MONOSPACE);
        report.setPadding(dp(8), dp(8), dp(8), dp(8));
        report.setTextIsSelectable(true);
        ScrollView scroll = new ScrollView(this);
        scroll.addView(report);
        page.addView(scroll, new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f));

        selectPackage = button("Selecionar arquivo XBLA");
        selectPackage.setOnClickListener(ignored -> choosePackage());
        page.addView(selectPackage);

        exportXex = button("Exportar default.xex");
        exportXex.setEnabled(false);
        exportXex.setOnClickListener(ignored -> exportDefaultXex());
        page.addView(exportXex);

        Button restart = button("Reiniciar runtime nativo");
        restart.setOnClickListener(ignored -> {
            runtimeReport = nativeStart(surface.getHolder().getSurface());
            showReports();
        });
        page.addView(restart);
        setContentView(page);
    }

    private void showReports() {
        report.setText(runtimeReport + "\n" + intakeReport);
    }

    private void choosePackage() {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("*/*");
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION |
                Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION);
        startActivityForResult(intent, PICK_XBLA);
    }

    private void inspectPackage(Uri uri) {
        selectPackage.setEnabled(false);
        exportXex.setEnabled(false);
        intakeReport = "[XBLA intake]\nReading package locally…\n";
        showReports();
        worker.execute(() -> {
            String result;
            try (ParcelFileDescriptor descriptor =
                         getContentResolver().openFileDescriptor(uri, "r")) {
                if (descriptor == null)
                    throw new IllegalStateException("document provider returned no file");
                if (privateXex.exists() && !privateXex.delete())
                    throw new IllegalStateException("cannot replace previous private XEX");
                int fd = descriptor.detachFd();
                result = nativeInspectPackage(fd, privateXex.getAbsolutePath(),
                        privateManifest.getAbsolutePath());
                if (result.contains("INTAKE_STATUS: PASS")) {
                    String sha256 = sha256(privateXex);
                    result += "default.xex SHA-256: " + sha256 + "\n";
                    try (FileOutputStream manifest = new FileOutputStream(privateManifest, true)) {
                        manifest.write(("\ndefault_xex_sha256=" + sha256 + "\n")
                                .getBytes(StandardCharsets.UTF_8));
                    }
                }
            } catch (Exception error) {
                result = "[XBLA intake]\nINTAKE_STATUS: FAIL\nReason: " +
                        error.getMessage() + "\n";
            }
            final String completed = result;
            runOnUiThread(() -> {
                intakeReport = completed;
                boolean ready = completed.contains("INTAKE_STATUS: PASS") && privateXex.isFile();
                exportXex.setEnabled(ready);
                selectPackage.setEnabled(true);
                showReports();
                Toast.makeText(this, ready ? "Case Zero verificado" : "Pacote rejeitado",
                        Toast.LENGTH_LONG).show();
            });
        });
    }

    private static String sha256(File file) throws Exception {
        MessageDigest digest = MessageDigest.getInstance("SHA-256");
        byte[] buffer = new byte[64 * 1024];
        try (FileInputStream input = new FileInputStream(file)) {
            int count;
            while ((count = input.read(buffer)) != -1) digest.update(buffer, 0, count);
        }
        StringBuilder value = new StringBuilder(64);
        for (byte item : digest.digest()) value.append(String.format("%02x", item & 0xff));
        return value.toString();
    }

    private void exportDefaultXex() {
        if (!privateXex.isFile()) {
            Toast.makeText(this, "Verifique o XBLA primeiro", Toast.LENGTH_LONG).show();
            return;
        }
        Intent intent = new Intent(Intent.ACTION_CREATE_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.setType("application/octet-stream");
        intent.putExtra(Intent.EXTRA_TITLE, "default.xex");
        startActivityForResult(intent, EXPORT_XEX);
    }

    private void writeExport(Uri destination) {
        worker.execute(() -> {
            String message;
            try (FileInputStream input = new FileInputStream(privateXex);
                 OutputStream output = getContentResolver().openOutputStream(destination, "wt")) {
                if (output == null)
                    throw new IllegalStateException("document provider returned no output");
                byte[] buffer = new byte[64 * 1024];
                int count;
                while ((count = input.read(buffer)) != -1) output.write(buffer, 0, count);
                message = "default.xex exportado com sucesso";
            } catch (Exception error) {
                message = "Falha ao exportar: " + error.getMessage();
            }
            final String completed = message;
            runOnUiThread(() -> Toast.makeText(this, completed, Toast.LENGTH_LONG).show());
        });
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode != RESULT_OK || data == null || data.getData() == null) return;
        Uri uri = data.getData();
        if (requestCode == PICK_XBLA) {
            int flags = data.getFlags() & Intent.FLAG_GRANT_READ_URI_PERMISSION;
            try {
                getContentResolver().takePersistableUriPermission(uri, flags);
            } catch (SecurityException ignored) {
                // Some providers grant access only for this process; inspection still works now.
            }
            getSharedPreferences(PREFS, MODE_PRIVATE).edit()
                    .putString(PREF_PACKAGE_URI, uri.toString()).apply();
            inspectPackage(uri);
        } else if (requestCode == EXPORT_XEX) {
            writeExport(uri);
        }
    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        runtimeReport = nativeStart(holder.getSurface());
        showReports();
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int width, int height) {
        // Swapchain resize handling belongs to the later first-frame stage.
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        runtimeReport = nativeStop();
        showReports();
    }

    @Override
    protected void onDestroy() {
        nativeStop();
        worker.shutdownNow();
        super.onDestroy();
    }
}
