use md_knowsearch::app;

fn main() {
    // 程序入口只负责启动应用层逻辑，具体命令处理放在 app 模块中。
    if let Err(error) = app::run() {
        // 命令行工具出错时打印错误并返回非零退出码。
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
