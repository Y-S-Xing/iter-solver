#[cfg(test)]
mod test {
    use std::fs;

    #[test]
    fn assert_readme() {
        let readme1 = fs::read_to_string("../README.md").expect("only use for whole workspace test");
        let readme2 = fs::read_to_string("README.md").expect("readme2");
        assert_eq!(readme1, readme2);
    }
}