# Edge-case registry

Generated from `edge_cases.yaml` by `make edge-doc`. 74 cases, 32 implemented and tested.

Status: **implemented** = code path exists and a test marked `@pytest.mark.edge("ID")` exercises it; **planned** = month-1 scope; **deferred** = tracked, not month 1.

## Ingest and time

| ID | Case | Trigger | Handling | Status | Tests |
|---|---|---|---|---|---|
| E-ING-01 | Camera clocks disagree | Cameras drift seconds apart; fusion hand-offs and the ±1 s fact-miner window fail. | Every timestamp is CamTime{cam_utc_ms, offset_ms, offset_confidence}; offsets estimated in the calibration walk and re-estimated from hand-offs; fusion uses corrected_ms() only. | implemented | test_schemas.py::test_camtime_offset_aligns_two_cameras |
| E-ING-02 | Stream drops or reconnects | RTSP disconnect mid-episode. | signal_lost/signal_restored events; open episode on that camera closes with status=truncated; tubes go to lost, not exited. | implemented | test_episode.py::test_signal_loss_truncates_episode |
| E-ING-03 | Variable or dropped frame rate | Camera throttles under load; fps changes. | Ticks are wall-clock windows (t_start,t_end), never frame indices; tube age computed from corrected time. | implemented | test_ingest.py::test_pts_filter_samples_to_target_fps<br>test_schemas.py::test_tick_is_wall_clock_window_not_frame_index |
| E-ING-04 | Codec without motion vectors | MJPEG or unknown codec stream. | Gate interface is codec-agnostic; FrameDiffGate fallback at 1 fps; camera profile records gate_mode. | implemented | test_ingest.py::test_codec_routes_to_gate_mode |
| E-ING-05 | Resolution or aspect change mid-stream | Camera reconfigured; homography and zones now invalid. | Reader SizeGuard flags the change; EventCompiler.on_size_change emits camera_moved_suspect; fusion freezes that camera until recalibration (week 3). | implemented | test_ingest.py::test_resolution_change_is_flagged_and_becomes_camera_moved_suspect |
| E-ING-06 | Timezone and DST | "Yesterday afternoon" across a DST switch or a site in another timezone. | All storage in UTC ms; site timezone stored in KB; agent resolves local windows explicitly and echoes them in the answer. | implemented | test_schemas.py::test_timestamps_are_utc_ms_only |
| E-ING-07 | Duplicate or out-of-order frames | RTSP jitter delivers repeated or reordered PTS. | Dedupe by PTS per camera; drop frames older than the last processed tick. | implemented | test_ingest.py::test_pts_filter_drops_duplicates_and_reordered_frames<br>test_ingest.py::test_reader_to_gate_on_synthetic_clip |

## Ring 0 · bitstream / motion gate

| ID | Case | Trigger | Handling | Status | Tests |
|---|---|---|---|---|---|
| E-GATE-01 | Stationary object is invisible to the gate | Person stops moving; bag placed and left; fallen person still. | Heartbeat detections on I-frames (1–2 s active tiles, 10–30 s quiet, always on asset homes) keep tubes alive; not the gate's job. | implemented | test_gate.py::test_stationary_object_fades_from_gate_by_design<br>test_tubes.py::test_heartbeat_detection_keeps_stationary_track_active |
| E-GATE-02 | Lighting change looks like whole-frame motion | Light switch, IR-cut toggle, cloud passes. | Global luminance step detected before differencing; emitted as illumination_change scene-state event; blobs suppressed for that update. | implemented | test_events.py::test_gate_results_become_scene_state_events<br>test_gate.py::test_light_switch_is_luma_step_not_motion<br>test_ingest.py::test_reader_to_gate_on_synthetic_clip |
| E-GATE-03 | Foliage, rain, insects, sensor noise | Persistent low-level motion that is not an object. | Adaptive noise floor (median block energy over history) × k; MV-field coherence test once MVGate lands. | implemented | test_gate.py::test_adaptive_noise_floor_ignores_sensor_noise_but_finds_object |
| E-GATE-04 | Camera shake or PTZ move | Most of the frame changes at once. | active_fraction > threshold => global_motion; blobs suppressed, background frozen; sustained => camera_moved_suspect. | implemented | test_gate.py::test_camera_shake_is_global_motion_and_suppresses_blobs |
| E-GATE-05 | Very slow motion under threshold | Someone creeping; object slid slowly. | Heartbeat detections catch state deltas the gate misses; gate FN rate is a tracked metric. | planned | — |
| E-GATE-06 | Motion in screens, mirrors, reflections | TV playing; mirror shows a person in another zone; steel reflects motion. | Scene card media_zones and reflective_surfaces become gate masks; tubes born inside them are reflection_suspect. | planned | — |
| E-GATE-07 | Intra-only or all-I-frame streams | Encoder configured with no P-frames; no motion vectors. | Same fallback as E-ING-04. | planned | — |
| E-GATE-08 | Night noise inflates motion energy | High-gain low-light mode. | Per-modality threshold profile; coherence weighted higher at night. | planned | — |

## Ring 1 · selective decode + detect

| ID | Case | Trigger | Handling | Status | Tests |
|---|---|---|---|---|---|
| E-DET-01 | Tiny objects below detector floor | Keys, phone, small tools at room-camera resolution. | Not detected as tubes; tracked via carried_item attribute and asset-home occupancy; inspect tool for on-demand native-res count/find; resolution ceiling stated in answers. | planned | — |
| E-DET-02 | Truncation at frame edge | Half a person at the border. | remap_detections flags boxes within 2 px of the frame border as truncated; Ring 2 then uses fallback_bbox_bottom with wide uncertainty and keyframe scoring penalises them. | implemented | test_detect_roi.py::test_remap_shifts_boxes_and_flags_frame_border_truncation |
| E-DET-03 | Class confusion | Child vs small adult; dog vs bag; mannequin vs person. | Class is an attribute with confidence; enrichment can override; agent never asserts class below threshold. | planned | — |
| E-DET-04 | Ghost detections in reflections | Mirror/glass shows a duplicate person. | Tubes inside reflective_surfaces are reflection_suspect; fusion never creates an entity from a suspect tube alone. | planned | — |
| E-DET-05 | People on screens, posters, photos | TV shows a face; framed photo detected as person. | media_zones from scene card; detections fully inside a media zone with no tube motion are dropped. | planned | — |
| E-DET-06 | Dense crowds | Heavy overlap; fragmentation. | crowd event on density; tube quality flag; enrichment skipped for low-quality tubes. | deferred | — |
| E-DET-07 | Fisheye or wide-lens distortion | Homography from a distorted image. | fov_class from scene card; undistort before foot-point projection. | deferred | — |
| E-DET-08 | IR appearance shift lowers recall | Night mode. | Measured on night eval set before any enhancer; enhancer only if gap is real and license clear. | planned | — |
| E-DET-09 | Static furniture and fixtures become tubes | Detector emits dining table, tv, chair as objects; tracker births permanent tubes for them. | Only TUBE_CLASSES (people, animals, vehicles, carried bags) become tubes; furniture is a scene-card asset owned by zones and heartbeats. | implemented | test_eval_mot.py::test_default_zones_and_tube_classes |

## Ring 2 · tubes and fusion

| ID | Case | Trigger | Handling | Status | Tests |
|---|---|---|---|---|---|
| E-TUBE-01 | ID switch when paths cross | Two people cross; IoU ambiguous. | Keep best match; record other as merge_candidate; never silently merge; fusion may later resolve with ReID. | implemented | test_tubes.py::test_crossing_paths_flag_merge_candidates_instead_of_guessing |
| E-TUBE-02 | Occlusion | Person behind furniture or another person. | state=occluded with occluded_since; after max_occluded_ms becomes lost (or exited if last box touched an exit zone). | implemented | test_tubes.py::test_occlusion_then_lost_or_exited |
| E-TUBE-03 | Long stationary dwell | Sleeping person; parked object. | Heartbeat detections with det_source=heartbeat keep the track active. | implemented | test_tubes.py::test_heartbeat_detection_keeps_stationary_track_active |
| E-TUBE-04 | Re-entry after leaving | Same person returns minutes later. | New tube; fusion links to the same entity via ReID + gallery; the agent reports entity, not tube. | planned | — |
| E-TUBE-05 | Same person in two overlapping cameras | Overlap region. | Per-camera tracker never merges; fusion merges by floor distance + ReID + time; enrichment once per entity, best view elected. | planned | — |
| E-TUBE-06 | Hand-off outside transit bounds | Entity appears in a non-adjacent tile too fast. | impossible_transition event; no merge; anomaly surfaced. | planned | — |
| E-TUBE-07 | Blind-spot ambiguity | Two enter a blind spot, one exits. | Delay-state keeps both candidates with probabilities; answer states ambiguity. | deferred | — |
| E-TUBE-08 | Appearance drift within a day | Jacket on/off; bag picked up. | Multiple exemplars per entity; embedding refresh on confident matches; oldest expire. | planned | — |
| E-TUBE-09 | Carried object is not its own tube | Bag on shoulder; keys in hand. | carried_item attribute on the person tube; object tube only when placed and stationary. | planned | — |
| E-TUBE-10 | Pets, strollers, wheelchairs | Non-person moving classes; child inside stroller. | Class-specific lifecycle; stroller+child handled as one tube with carried_item=child hint. | deferred | — |
| E-TUBE-13 | Fragmentation at sampled frame rate | Decode at 2 fps means a walking person moves half a box width between ticks; IoU association breaks the tube. | dt-aware Kalman prediction (ByteTrack rewrite) and a lower IoU gate; measured as fragmentation_ratio and idsw in bench/ring2_tubes.py against the SimpleIoUTracker baseline. | planned | — |
| E-TUBE-12 | Feet occluded, foot point wrong | Person behind counter. | FloorPoint.source=fallback_bbox_bottom and uncertainty widened; fusion tolerances read it. | implemented | test_tubes.py::test_occluded_feet_widen_uncertainty_and_tag_source |

## Ring 3a · event compiler

| ID | Case | Trigger | Handling | Status | Tests |
|---|---|---|---|---|---|
| E-EVT-01 | Zone boundary jitter | Foot point oscillates on a zone edge. | Hysteresis: enter after N consecutive inside ticks, exit after N outside. | implemented | test_events.py::test_zone_hysteresis_filters_boundary_jitter |
| E-EVT-02 | Cross-camera event order under clock offset | Kitchen camera 2 s ahead of hallway. | Sort by corrected_ms; hand-off window tolerates offset_confidence. | planned | — |
| E-EVT-03 | False pickup from occlusion of the shelf | Person stands in front of the asset home; heartbeat cannot see the asset. | Pickup requires asset absent on a heartbeat taken with nobody inside the zone, after a present reading; subject = visitors in between; no visitors => asset_missing_from_home with no blame. | implemented | test_events.py::test_pickup_requires_absence_with_nobody_in_zone<br>test_events.py::test_missing_asset_with_no_visitor_does_not_blame_anyone |
| E-EVT-04 | Fall vs lying down on purpose | Person lies on sofa/bed. | rest zones from scene card suppress fall; fall requires vertical velocity + pose primitive outside rest zones. | deferred | — |
| E-EVT-05 | Left-behind vs placed at home | Bag set on its usual hook. | left_behind only outside the object's home zone. | planned | — |
| E-EVT-06 | Duplicate events from overlapping cameras | Both cameras see the same enter. | dedupe_key = type|entity|time bucket at fusion; per-camera events keep camera_id for evidence. | implemented | test_events.py::test_overlapping_cameras_share_dedupe_key_but_not_event_id |
| E-EVT-07 | Episode never closes | Busy lobby active for hours. | soft_cut on cast churn > threshold or max duration; events may span episodes. | implemented | test_episode.py::test_soft_cut_on_cast_churn_or_max_duration |
| E-EVT-08 | Event spans an episode boundary | Dwell starts in one episode and ends in the next. | Events carry episode_id of emission; queries join across episodes by entity and time, never by episode alone. | planned | — |

## Ring 3b · foveation and enrichment

| ID | Case | Trigger | Handling | Status | Tests |
|---|---|---|---|---|---|
| E-FOV-01 | Crop too small to describe | Short side < 96 px. | Enhancement ladder; enhanced=true propagates to attributes; confidence discounted. | implemented | test_schemas.py::test_enhanced_attributes_are_discounted |
| E-FOV-02 | Motion blur | Fast movement. | Keyframe score uses Laplacian sharpness; blurred frames skipped; if none sharp, attributes low confidence. | planned | — |
| E-FOV-03 | Backlit or overexposed subject | Doorway against daylight. | Exposure score in keyframe selection; try other keyframes; color_reason=low_confidence. | planned | — |
| E-FOV-04 | Cross-cell attribute bleed in contact sheets | VLM describes cell 3 with cell 4's jacket. | Hard borders + cell ids; fixed expected_cells; per-cell schema; bleed rate measured on eval set; fall back to 12 cells if >5%. | implemented | test_schemas.py::test_contact_sheet_rejects_duplicate_missing_or_mismatched_cells |
| E-FOV-05 | Hallucinated color in IR | Monochrome crop. | Attributes validator and grammar forbid colors when modality is ir/thermal; color_reason=ir_mode; tone fields instead. | implemented | test_schemas.py::test_ir_forbids_color_attributes |
| E-FOV-06 | Ring buffer expired before enrichment | Slow path lags past buffer window. | Best keyframe crop persisted to R2 at tube birth; keyframe_refs on the tube. | implemented | test_schemas.py::test_tube_carries_keyframe_refs<br>test_tubes.py::test_keyframe_ref_persisted_at_birth |
| E-FOV-07 | VLM output violates schema | Grammar bug or truncation. | Retry once; then EnrichmentPatch{failed=true}; tick never blocked. | implemented | test_schemas.py::test_bad_vlm_output_returns_none_and_failed_patch_is_valid |
| E-FOV-08 | Attributes contradict across keyframes | Red jacket in one, dark in another. | Majority vote over keyframes; keep history; confidence reflects agreement. | planned | — |

## Storage and handoff

| ID | Case | Trigger | Handling | Status | Tests |
|---|---|---|---|---|---|
| E-STO-01 | Patch arrives after episode closed | Slow path finishes late. | Episode file is append-only; PatchRecord accepted after close. | implemented | test_episode.py::test_patch_after_close_is_appended |
| E-STO-02 | Retries create duplicates | Writer crashes and replays. | Deterministic episode/event ids; per-record idempotency keys. | implemented | test_episode.py::test_writes_are_idempotent |
| E-STO-03 | Retention expired for a clip | Clip requested for footage past retention. | clip tool checks availability; degrades to keyframes; answer states retention. | planned | — |
| E-STO-04 | Schema version bump | New field added mid-deployment. | Provenance.schema_version on every record; readers tolerate unknown fields; migration script per bump. | implemented | test_schemas.py::test_provenance_required_and_unknown_fields_tolerated |
| E-STO-05 | KB version drift | Transit bounds changed after an episode was compiled. | Provenance.kb_version on every record; replays pin the version. | implemented | test_schemas.py::test_provenance_required_and_unknown_fields_tolerated |

## Block 2 · agent

| ID | Case | Trigger | Handling | Status | Tests |
|---|---|---|---|---|---|
| E-AGT-01 | Ambiguous anchor | "After Jay went out" matches two exits. | One clarifying question; never a silent pick. | planned | — |
| E-AGT-02 | No results | Nothing matches the window. | Say so; propose widening; never fabricate. | planned | — |
| E-AGT-03 | Unknown named entity | "Jay" not in gallery. | Ask who Jay is; offer naming form. | planned | — |
| E-AGT-04 | Fuzzy time expressions | "yesterday afternoon", "after lunch". | Resolve to explicit local window; echo it in the answer. | planned | — |
| E-AGT-05 | Color question on IR footage | "what color was his shirt" at night. | State IR mode; give tone; never a color. | planned | — |
| E-AGT-06 | Unverified claim | Attribute from a low-confidence patch. | verify() before asserting; every claim cites record ids and timestamps. | planned | — |
| E-AGT-07 | Broken custody chain | carried_item missed; drop never fired. | Graceful degrade: last known custody + offer clip. | planned | — |
| E-AGT-08 | Duplicates from overlapping cameras in the answer | Same person listed twice. | Answer over entities, not tubes. | planned | — |

## Knowledge base and calibration

| ID | Case | Trigger | Handling | Status | Tests |
|---|---|---|---|---|---|
| E-KB-01 | Camera moved after calibration | Bumped or remounted. | Sustained global motion => camera_moved_suspect; landmark check on I-frames; freeze fusion; prompt recalibration. | implemented | test_events.py::test_gate_results_become_scene_state_events |
| E-KB-02 | Scene card labels wrong | "Dining hall" is actually the kitchen. | Everything on the card is a hypothesis; user correction bumps version and re-labels. | planned | — |
| E-KB-03 | Causal fact contradicted | Switch flipped, light did not change. | Fact.add_contradiction; retire at 2 unless user_confirmed. | implemented | test_fact.py::test_facts_promote_on_support_and_retire_on_contradiction<br>test_fact.py::test_user_confirmed_facts_survive_contradictions |
| E-KB-04 | Furniture moved | Shelf relocated; asset home zone stale. | Repeated asset_missing with asset detected elsewhere => propose new home. | deferred | — |

## Night modality

| ID | Case | Trigger | Handling | Status | Tests |
|---|---|---|---|---|---|
| E-NIGHT-01 | IR switch mid-tube | Camera toggles IR while a person is in view. | modality_switch event; tube segments carry modality; attributes per segment. | planned | — |
| E-NIGHT-02 | ReID does not transfer day to IR | Same person, different modality. | Separate exemplar sets per modality; tile/time continuity weighted higher at night. | planned | — |
| E-NIGHT-03 | IR blooming near camera | Subject washed out by illuminator. | Exposure score rejects keyframe; low confidence attributes. | deferred | — |

## Privacy

| ID | Case | Trigger | Handling | Status | Tests |
|---|---|---|---|---|---|
| E-PRIV-01 | Gallery entry deletion | User deletes a named person. | Cascade: exemplars, names on records, KB facts referencing the entity; anonymous ids remain. | planned | — |
| E-PRIV-02 | Face embeddings without opt-in | Face model enabled by default. | Site-level opt-in flag gates the face lane; default off. | planned | — |
