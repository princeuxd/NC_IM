"""Streamlit UI for creator onboarding.

This module contains the UI-only logic for managing and onboarding YouTube
creators. All heavy-lifting (file IO, OAuth interactions, etc.) is delegated to
`helpers.creators`.
"""
from __future__ import annotations

import os
import time

import streamlit as st

from src.helpers import creators as hc

# Re-export constant so callers can use it if they wish
TOKENS_DIR = hc.TOKENS_DIR


def _format_number(num: int) -> str:
    """Human-friendly formatting for subscriber / video counts."""
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    if num >= 1_000:
        return f"{num/1_000:.1f}K"
    return str(num)


def render_onboarding() -> None:
    """Render the Streamlit UI for creator management & onboarding."""

    # Page header with stats
    col_header, col_stats = st.columns([3, 1])
    with col_header:
        st.title("🎯 Creator Management Hub")
        st.markdown(
            "Manage YouTube creator OAuth credentials and monitor authentication status"
        )

    with col_stats:
        token_files = hc.list_token_files()
        st.metric("🔑 Active Creators", len(token_files))

    tab_creators, tab_onboard = st.tabs(["👥 Manage Creators", "➕ Add New Creator"])

    # ------------------------------------------------------------------
    # Manage Creators
    # ------------------------------------------------------------------
    with tab_creators:
        if not token_files:
            st.info(
                "🚀 **Get started** by adding your first creator in the 'Add New Creator' tab!"
            )
        else:
            st.subheader("Active Creator Accounts")

            col_batch1, col_batch2, _ = st.columns([1, 1, 2])
            with col_batch1:
                if st.button("🔄 Refresh All", help="Refresh all expired tokens"):
                    refreshed = 0
                    for tf in token_files:
                        details = hc.get_creator_details(tf)
                        if not details["is_valid"] and hc.refresh_creator_token(
                            details["channel_id"]
                        ):
                            refreshed += 1
                    if refreshed:
                        st.success(f"Refreshed {refreshed} creator token(s)")
                        st.rerun()
                    else:
                        st.info("No tokens needed refreshing")

            with col_batch2:
                if st.button("📊 Export List", help="Export creator list to JSON"):
                    import json

                    creator_list = [
                        {
                            "channel_id": det["channel_id"],
                            "title": det["title"],
                            "is_valid": det["is_valid"],
                            "last_checked": det["last_checked"],
                        }
                        for det in (hc.get_creator_details(tf) for tf in token_files)
                    ]
                    st.download_button(
                        "💾 Download creators.json",
                        json.dumps(creator_list, indent=2),
                        "creators.json",
                        "application/json",
                    )

            st.divider()

            for tf in token_files:
                details = hc.get_creator_details(tf)

                status_color = "🟢" if details["is_valid"] else "🔴"
                status_text = "Active" if details["is_valid"] else "Invalid/Expired"

                with st.container():
                    col_avatar, col_info, col_stats, col_actions = st.columns(
                        [1, 3, 2, 2]
                    )

                    with col_avatar:
                        if thumb := details.get("thumbnail_url"):
                            try:
                                st.image(thumb, width=60)
                            except Exception:
                                st.markdown("👤")
                        else:
                            st.markdown("👤")

                    with col_info:
                        st.markdown(f"**{details['title']}**")
                        st.caption(f"{status_color} {status_text}")
                        st.caption(f"ID: `{details['channel_id']}`")

                    with col_stats:
                        if details["is_valid"]:
                            st.metric("👥 Subscribers", _format_number(details["subscriber_count"]))
                            st.caption(f"📹 {_format_number(details['video_count'])} videos")
                        else:
                            st.markdown("⚠️ **Token Invalid**")
                            if err := details.get("error"):
                                st.caption(f"Error: {err}")

                    with col_actions:
                        col_a1, col_a2 = st.columns(2)

                        with col_a1:
                            if not details["is_valid"]:
                                if st.button(
                                    "🔄",
                                    key=f"refresh_{details['channel_id']}",
                                    help="Refresh token",
                                ):
                                    if hc.refresh_creator_token(details["channel_id"]):
                                        st.success("Token refreshed!")
                                        st.rerun()
                                    else:
                                        st.error("Failed to refresh token")
                            else:
                                st.button("✅", disabled=True, help="Token is valid")

                        with col_a2:
                            if st.button(
                                "🗑️",
                                key=f"remove_{details['channel_id']}",
                                help="Remove creator",
                            ):
                                confirm_key = f"confirm_{details['channel_id']}"
                                if not st.session_state.get(confirm_key):
                                    st.session_state[confirm_key] = True
                                    st.warning(
                                        f"⚠️ Click again to confirm removal of **{details['title']}**"
                                    )
                                else:
                                    if hc.remove_creator(details["channel_id"]):
                                        st.success("Creator removed")
                                        st.session_state.pop(confirm_key, None)
                                        st.rerun()
                                    else:
                                        st.error("Failed to remove creator")

                st.divider()

    # ------------------------------------------------------------------
    # Add New Creator (OAuth flow)
    # ------------------------------------------------------------------
    with tab_onboard:
        st.subheader("🚀 Add New Creator")
        st.markdown("Connect a YouTube creator account using OAuth 2.0 authentication")

        st.markdown("### Step 1: OAuth Configuration")
        env_cfg = hc.validate_env_oauth_config()

        col_cfg, col_val = st.columns([2, 1])
        with col_cfg:
            st.markdown("**Required Environment Variables:**")

            cid, csecret, pid = (
                os.getenv("OAUTH_CLIENT_ID"),
                os.getenv("OAUTH_CLIENT_SECRET"),
                os.getenv("OAUTH_PROJECT_ID"),
            )
            id_status = "✅" if cid else "❌"
            secret_status = "✅" if csecret else "❌"
            project_status = "✅" if pid else "⚠️"

            st.code(
                f"""
{id_status} OAUTH_CLIENT_ID={'Set' if cid else 'Missing'}
{secret_status} OAUTH_CLIENT_SECRET={'Set' if csecret else 'Missing'}
{project_status} OAUTH_PROJECT_ID={'Set' if pid else 'Optional (default)'}
                """
            )

            if not env_cfg["valid"]:
                st.error("❌ Please set the required environment variables in your `.env` file.")
                st.code(
                    """
# Add these to your .env file:
OAUTH_CLIENT_ID=your_client_id_here
OAUTH_CLIENT_SECRET=your_client_secret_here
OAUTH_PROJECT_ID=your_project_id_here  # Optional
                    """
                )

                with st.expander("🔧 How to get these values"):
                    st.markdown(
                        """
1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create or select a project
3. Enable the YouTube Data API v3 and YouTube Analytics API
4. Navigate to **Credentials** → **Create Credentials** → **OAuth 2.0 Client IDs**
5. Choose **Desktop application** and download the JSON file.
                        """
                    )

        with col_val:
            if env_cfg["valid"]:
                st.success("✅ OAuth Config Valid")
                st.caption(f"🏗️ Project: {env_cfg.get('project_id', 'Default')}")
                st.caption(f"👤 Client ID: ...{env_cfg.get('client_id', '')[-8:]}")
                st.caption(f"🔧 Type: {env_cfg.get('type', 'installed')}")
            else:
                st.error("❌ Configuration Invalid")
                st.caption(env_cfg.get("error", "Unknown error"))

        if not env_cfg["valid"]:
            return  # Halt UI until config is fixed

        st.markdown("### Step 2: OAuth Authentication")
        col_oauth, col_info = st.columns([1, 2])
        with col_oauth:
            oauth_btn = st.button(
                "🔐 Start OAuth Flow",
                key="oauth_start",
                type="primary",
                help="Opens Google OAuth consent screen in a new tab",
            )

        if oauth_btn and not st.session_state.get("oauth_flow_active"):
            st.session_state["oauth_flow_active"] = True

        with col_info:
            st.info(
                """
**What happens next:**
1. 🌐 Google OAuth page opens in new tab
2. 📋 Select your YouTube channel
3. ✅ Grant required permissions
4. 🎉 Channel gets added to your dashboard
                """
            )

        if st.session_state.get("oauth_flow_active"):
            with st.spinner(
                "🔄 Initiating OAuth flow... Please complete authentication in the new browser tab."
            ):
                try:
                    bar = st.progress(0)
                    txt = st.empty()

                    txt.text("⏳ Creating OAuth configuration...")
                    bar.progress(25)
                    time.sleep(0.5)

                    tmp_secret = hc.create_temp_client_secret_file()
                    if not tmp_secret:
                        raise RuntimeError(
                            "Failed to create OAuth configuration from environment variables"
                        )

                    txt.text("🔐 Opening OAuth consent screen...")
                    bar.progress(50)

                    token_path, cid_ret, title = hc.onboard_creator(tmp_secret)

                    bar.progress(75)
                    txt.text("📊 Fetching channel information...")
                    time.sleep(0.5)

                    bar.progress(100)
                    txt.text("✅ Success!")

                    if tmp_secret.exists():
                        tmp_secret.unlink()

                    st.balloons()
                    st.success(
                        f"""
🎉 **Successfully onboarded!**

**Channel:** {title}  
**ID:** `{cid_ret}`  
**Token:** `{token_path.name}`
                        """
                    )

                    st.session_state.pop("oauth_flow_active", None)
                    st.info(
                        "💡 Switch to the 'Manage Creators' tab to see your new creator account!"
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    st.error(
                        f"""
❌ **OAuth onboarding failed**

**Error:** {str(exc)}
                        """
                    )
                    if (TOKENS_DIR / "_temp_client_secret.json").exists():
                        (TOKENS_DIR / "_temp_client_secret.json").unlink()
                    st.session_state.pop("oauth_flow_active", None)
                    with st.expander("🔧 Debug Information"):
                        st.code(f"Error Type: {type(exc).__name__}\nError Message: {exc}")
