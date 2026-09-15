package com.risingrecomp.diagnostic;

import android.app.Activity;
import android.content.Intent;
import android.content.SharedPreferences;
import android.graphics.Color;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.DocumentsContract;
import android.text.TextUtils;
import android.view.Gravity;
import android.view.View;
import android.widget.Button;
import android.widget.LinearLayout;
import android.widget.ScrollView;
import android.widget.TextView;
import android.widget.Toast;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.InputStreamReader;
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.text.SimpleDateFormat;
import java.util.Date;
import java.util.Locale;

public final class MainActivity extends Activity {
    private static final int PICK_LOG_FOLDER = 1001;
    private static final String PREFS = "risingrecomp_diagnostic";
    private static final String PREF_LOG_FOLDER = "log_folder";

    static {
        System.loadLibrary("rising_diagnostic");
    }

    private TextView output;
    private File logFile;
    private boolean exportAfterFolderPick;

    private static native String collectNativeDiagnostics();

    @Override
    protected void onCreate(Bundle state) {
        super.onCreate(state);
        buildInterface();
        logFile = new File(new File(getFilesDir(), "logs"), "risingrecomp-diagnostic.log");
        runDiagnostics();
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

    private Button button(String label, View.OnClickListener listener) {
        Button button = new Button(this);
        button.setText(label);
        button.setAllCaps(false);
        button.setOnClickListener(listener);
        LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT);
        params.topMargin = dp(8);
        button.setLayoutParams(params);
        return button;
    }

    private void buildInterface() {
        LinearLayout page = new LinearLayout(this);
        page.setOrientation(LinearLayout.VERTICAL);
        page.setPadding(dp(20), dp(24), dp(20), dp(20));
        page.setBackgroundColor(Color.rgb(255, 251, 254));

        TextView title = text("RisingRecomp", 28, Color.rgb(35, 31, 32));
        title.setGravity(Gravity.START);
        page.addView(title);

        TextView subtitle = text("Android compatibility diagnostic", 15, Color.rgb(90, 85, 90));
        subtitle.setPadding(0, dp(2), 0, dp(14));
        page.addView(subtitle);

        output = text("Collecting diagnostics…", 13, Color.rgb(40, 40, 40));
        output.setTextIsSelectable(true);
        output.setTypeface(android.graphics.Typeface.MONOSPACE);
        output.setPadding(dp(12), dp(12), dp(12), dp(12));
        output.setBackgroundColor(Color.rgb(244, 239, 246));

        ScrollView scroll = new ScrollView(this);
        scroll.addView(output);
        LinearLayout.LayoutParams scrollParams = new LinearLayout.LayoutParams(
                LinearLayout.LayoutParams.MATCH_PARENT, 0, 1f);
        scroll.setLayoutParams(scrollParams);
        page.addView(scroll);

        page.addView(button("Ver logs", ignored -> showLog()));
        page.addView(button("Exportar logs", ignored -> exportLog()));
        page.addView(button("Logs folder", ignored -> chooseLogFolder(false)));

        setContentView(page);
    }

    private void runDiagnostics() {
        StringBuilder report = new StringBuilder();
        report.append("RisingRecomp compatibility probe 0.2.0\n");
        report.append("Timestamp: ").append(new Date()).append('\n');
        report.append("Manufacturer: ").append(Build.MANUFACTURER).append('\n');
        report.append("Model: ").append(Build.MODEL).append('\n');
        report.append("Android: ").append(Build.VERSION.RELEASE)
                .append(" (API ").append(Build.VERSION.SDK_INT).append(")\n");
        report.append("ABIs: ").append(TextUtils.join(", ", Build.SUPPORTED_ABIS)).append('\n');
        report.append("Runtime max memory: ")
                .append(Runtime.getRuntime().maxMemory() / (1024 * 1024)).append(" MiB\n");
        report.append("Storage free: ")
                .append(getFilesDir().getFreeSpace() / (1024 * 1024)).append(" MiB\n");
        report.append("External storage: ").append(Environment.getExternalStorageState()).append("\n\n");
        report.append(collectNativeDiagnostics());

        String value = report.toString();
        output.setText(value);
        if (!logFile.getParentFile().exists() && !logFile.getParentFile().mkdirs()) {
            toast("Não foi possível criar a pasta interna de logs");
            return;
        }
        try (FileOutputStream stream = new FileOutputStream(logFile, false)) {
            stream.write(value.getBytes(StandardCharsets.UTF_8));
        } catch (Exception error) {
            toast("Falha ao salvar log: " + error.getMessage());
        }
    }

    private void showLog() {
        if (!logFile.isFile()) {
            toast("Nenhum log disponível");
            return;
        }
        StringBuilder content = new StringBuilder();
        try (BufferedReader reader = new BufferedReader(new InputStreamReader(
                new FileInputStream(logFile), StandardCharsets.UTF_8))) {
            String line;
            while ((line = reader.readLine()) != null) {
                content.append(line).append('\n');
            }
            output.setText(content.toString());
        } catch (Exception error) {
            toast("Falha ao ler log: " + error.getMessage());
        }
    }

    private void chooseLogFolder(boolean exportAfterPick) {
        exportAfterFolderPick = exportAfterPick;
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT_TREE);
        intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION
                | Intent.FLAG_GRANT_WRITE_URI_PERMISSION
                | Intent.FLAG_GRANT_PERSISTABLE_URI_PERMISSION);
        startActivityForResult(intent, PICK_LOG_FOLDER);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != PICK_LOG_FOLDER || resultCode != RESULT_OK || data == null) {
            exportAfterFolderPick = false;
            return;
        }
        Uri uri = data.getData();
        if (uri == null) return;
        int flags = data.getFlags() & (Intent.FLAG_GRANT_READ_URI_PERMISSION
                | Intent.FLAG_GRANT_WRITE_URI_PERMISSION);
        getContentResolver().takePersistableUriPermission(uri, flags);
        getSharedPreferences(PREFS, MODE_PRIVATE).edit()
                .putString(PREF_LOG_FOLDER, uri.toString()).apply();
        toast("Pasta de logs selecionada");
        if (exportAfterFolderPick) exportTo(uri);
        exportAfterFolderPick = false;
    }

    private void exportLog() {
        SharedPreferences prefs = getSharedPreferences(PREFS, MODE_PRIVATE);
        String stored = prefs.getString(PREF_LOG_FOLDER, null);
        if (stored == null) {
            chooseLogFolder(true);
        } else {
            exportTo(Uri.parse(stored));
        }
    }

    private void exportTo(Uri treeUri) {
        if (!logFile.isFile()) {
            toast("Nenhum log disponível");
            return;
        }
        try {
            Uri parent = DocumentsContract.buildDocumentUriUsingTree(
                    treeUri, DocumentsContract.getTreeDocumentId(treeUri));
            String stamp = new SimpleDateFormat("yyyyMMdd-HHmmss", Locale.US).format(new Date());
            Uri destination = DocumentsContract.createDocument(getContentResolver(), parent,
                    "text/plain", "RisingRecomp-Diagnostic-" + stamp + ".txt");
            if (destination == null) throw new IllegalStateException("destination unavailable");
            try (FileInputStream input = new FileInputStream(logFile);
                 OutputStream outputStream = getContentResolver().openOutputStream(destination)) {
                if (outputStream == null) throw new IllegalStateException("output unavailable");
                byte[] buffer = new byte[8192];
                int count;
                while ((count = input.read(buffer)) != -1) outputStream.write(buffer, 0, count);
            }
            toast("Log exportado");
        } catch (Exception error) {
            toast("Falha ao exportar: " + error.getMessage());
        }
    }

    private void toast(String message) {
        Toast.makeText(this, message, Toast.LENGTH_LONG).show();
    }
}
