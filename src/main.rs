use md_knowsearch::app;

fn main() {
    if let Err(error) = app::run() {
        eprintln!("error: {error}");
        std::process::exit(1);
    }
}
