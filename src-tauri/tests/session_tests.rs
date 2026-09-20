use std::time::Duration;

use mlp_server::session::{validate_session_id, SessionStore};

#[tokio::test]
async fn sessions_are_isolated() {
    let store = SessionStore::new(Duration::from_secs(3600), 1000);
    let alpha = store.get_or_create("alpha").await.unwrap();
    let beta = store.get_or_create("beta").await.unwrap();

    {
        let mut alpha = alpha.lock().await;
        alpha.train_steps(1).unwrap();
    }

    assert_eq!(alpha.lock().await.response("alpha").step_count, 1);
    assert_eq!(beta.lock().await.response("beta").step_count, 0);
}

#[test]
fn session_ids_reject_unsafe_values() {
    assert!(validate_session_id("abc-123_DEF").is_ok());
    assert!(validate_session_id("").is_err());
    assert!(validate_session_id("contains space").is_err());
    assert!(validate_session_id(&"a".repeat(65)).is_err());
}
