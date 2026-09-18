WEAVE ARCHITECTURE PROPRIETARY LICENSE

Version 1.0 — September 18, 2026

Copyright © 2026 Thomas Price. All rights reserved.



PREAMBLE

This License governs the use, reproduction, modification, and distribution of the Weave Architecture, including all of its architectural components, algorithms, training strategies, pseudocode, specifications, and derivative works, as originally designed and authored by Thomas Price ("the Author"). This License is legally binding upon all parties who access, reference, implement, or otherwise make use of any portion of the Weave Architecture.



SECTION 1 — DEFINITIONS

1.1 "Weave Architecture" refers to the complete 5-billion-parameter transformer system designed by Thomas Price, encompassing all components described in the Weave Architecture Blueprint Whitepaper v1.0, including but not limited to: WeaveTransformer, WeaveBlock, multi-branch residual streams (Branch A, Branch B, Branch C), WeaveMerge, WeaveGrid, GridNode, cross-grid edges, chunked grouping attention, StabilityCluster, ZoomProfile, LatticeProfile, LayerLattice, collapse\_with\_zoom\_and\_lattice, forward\_dynamic, and the CPU-centric micro-chunk training strategy.



1.2 "Weave Core" refers exclusively to the multi-branch residual weaving subsystem, comprising: multi-branch residual streams (Branch A, Branch B, Branch C), the WeaveMerge mechanism, and the weave branch projection and gating logic. The Weave Core does NOT include: WeaveGrid, GridNode, cross-grid edges, chunked grouping, StabilityCluster, ZoomProfile, LatticeProfile, LayerLattice, collapse\_with\_zoom\_and\_lattice, forward\_dynamic, or the micro-chunk training strategy.



1.3 "Full Architecture" means the complete Weave Architecture as defined in Section 1.1.



1.4 "Open Source Model" means any trained model, checkpoint, or derivative model whose weights, training code, architecture code, training data sources, and documentation are publicly released under a recognized open-source license (such as Apache 2.0, MIT, or a comparable open license approved by the Open Source Initiative), and made freely and unconditionally available to the general public without access restrictions, paywalls, or usage fees.



1.5 "AMD" means Advanced Micro Devices, Inc., its wholly owned subsidiaries, and any entities directly controlled by Advanced Micro Devices, Inc.



1.6 "NVIDIA" means NVIDIA Corporation, its wholly owned subsidiaries, and any entities directly controlled by NVIDIA Corporation.



1.7 "Commercial Use" means any use intended for or resulting in commercial advantage, monetary compensation, or integration into a commercial product or service.



1.8 "Derivative Work" means any work that incorporates, adapts, modifies, extends, or is architecturally derived from any component of the Weave Architecture or Weave Core.



SECTION 2 — GRANT OF RIGHTS TO AMD

2.1 Conditional Full License.

Subject to full compliance with all conditions stated in Section 2.2, the Author hereby grants to AMD a non-exclusive, non-transferable, worldwide license to:



(a) Use, implement, reproduce, and deploy the Full Architecture;



(b) Modify or extend the Full Architecture to create Derivative Works;



(c) Train models using the CPU-centric micro-chunk training strategy and all associated training components;



(d) Integrate the Full Architecture into AMD hardware, software, research platforms, and AI frameworks.



2.2 Conditions — AMD License is VOID if ANY of the following are violated:



(a) Open Source Obligation: AMD must release, as an Open Source Model (as defined in Section 1.4), every model trained using the Full Architecture or any substantial portion thereof. This obligation applies to all model weights, training scripts, configuration files, and architecture implementations derived from the Weave Architecture. No model trained on or derived from the Full Architecture may be withheld from the public, offered under a restricted license, or deployed as a closed proprietary product.



(b) Attribution: AMD must clearly credit Thomas Price as the original designer and author of the Weave Architecture in all public releases, publications, model cards, documentation, and marketing materials that reference the architecture or any Derivative Work thereof. Attribution must appear prominently and must not be obscured, minimized, or omitted.



(c) No Sublicensing: AMD may not sublicense any rights granted under this Section to any third party, including AMD's customers, partners, or subsidiary entities not wholly owned by AMD.



(d) No Weaponization: AMD may not use the Full Architecture or any Derivative Work thereof in any system designed for autonomous weapons, lethal decision-making systems, mass surveillance, or any application that violates international humanitarian law.



(e) License Passthrough: Any Derivative Work released by AMD must be accompanied by this License or a license no more restrictive than this License, and must preserve all conditions of Section 2.2 for any downstream recipients.



2.3 Termination of AMD License.

AMD's license under this Section terminates automatically and immediately, without notice, upon any breach of Section 2.2. Upon termination, AMD must cease all use of the Full Architecture and destroy all implementations, trained weights, and Derivative Works derived from the Full Architecture within sixty (60) days.



SECTION 3 — GRANT OF RIGHTS TO NVIDIA

3.1 Restricted Weave Core License.

The Author hereby grants to NVIDIA a non-exclusive, non-transferable, worldwide license to:



(a) Use and implement the Weave Core only (as defined in Section 1.2);



(b) Integrate the Weave Core into NVIDIA research, software frameworks, and AI systems.



3.2 Explicit Restrictions — NVIDIA:



(a) Full Architecture Prohibited: NVIDIA is expressly prohibited from using, implementing, reproducing, referencing, or deriving any Derivative Work from any component of the Full Architecture beyond the Weave Core. This prohibition includes, without limitation: WeaveGrid, GridNode, cross-grid edges, chunked grouping attention, StabilityCluster, ZoomProfile, LatticeProfile, LayerLattice, collapse\_with\_zoom\_and\_lattice, forward\_dynamic, and the micro-chunk training strategy.



(b) No Combination: NVIDIA may not implement the Weave Core in combination with independently developed systems that replicate the function or behavior of any restricted component listed in Section 3.2(a), even if those systems are differently named or described.



(c) Attribution Required: NVIDIA must clearly credit Thomas Price as the original designer of the Weave Core in all publications, model cards, documentation, software releases, and marketing materials that reference the Weave Core or any Derivative Work thereof.



(d) No Sublicensing: NVIDIA may not sublicense any rights granted under this Section to any third party.



(e) No Weaponization: NVIDIA may not use the Weave Core in any system designed for autonomous weapons, lethal decision-making systems, mass surveillance, or any application that violates international humanitarian law.



3.3 Termination of NVIDIA License.

NVIDIA's license under this Section terminates automatically and immediately, without notice, upon any breach of Section 3.2. Upon termination, NVIDIA must cease all use of the Weave Core and destroy all implementations and Derivative Works within sixty (60) days.



SECTION 4 — ALL OTHER PARTIES

4.1 No Grant.

No license, right, permission, or interest of any kind is granted to any party other than AMD and NVIDIA under this License. All other individuals, corporations, research institutions, governments, and entities are expressly prohibited from using, copying, implementing, reproducing, modifying, distributing, sublicensing, or creating Derivative Works of the Full Architecture or any component thereof without express written permission from Thomas Price.



4.2 Research Exception.

Notwithstanding Section 4.1, academic researchers at accredited, non-profit educational institutions may reference the Weave Architecture in published academic papers, conduct theoretical analysis, and reproduce pseudocode or diagrams in academic publications, provided that:



(a) No implementation or trained model is produced;



(b) Thomas Price is credited as the original author in all publications;



(c) The work is not funded by, performed on behalf of, or transferred to AMD, NVIDIA, or any commercial entity.



SECTION 5 — INTELLECTUAL PROPERTY \& OWNERSHIP

5.1 Retained Ownership.

Thomas Price retains full, exclusive copyright and intellectual property ownership over the Full Architecture, the Weave Core, and all associated components, algorithms, pseudocode, and specifications, in all jurisdictions worldwide. Nothing in this License transfers, assigns, or diminishes the Author's ownership rights.



5.2 No Implied Rights.

No rights are granted by implication, estoppel, or otherwise beyond those expressly stated in this License.



5.3 Patent Rights.

This License does not grant any patent license. If any claim in any patent application, whether pending or issued, covers any component of the Weave Architecture, no license to that patent is granted under this License without a separate written agreement signed by Thomas Price.



SECTION 6 — ENFORCEMENT \& REMEDIES

6.1 Injunctive Relief.

The Author reserves the right to seek immediate injunctive relief in any court of competent jurisdiction to halt any unauthorized use, implementation, or distribution of the Weave Architecture or Weave Core, without the requirement of posting a bond.



6.2 Damages.

Any party that violates this License is liable for: (a) all actual damages suffered by the Author; (b) all profits attributable to the unauthorized use; (c) statutory damages where applicable; and (d) the Author's reasonable attorneys' fees and litigation costs.



6.3 Audit Rights — AMD.

The Author reserves the right to request, no more than once per calendar year, written confirmation from AMD that all models trained using the Full Architecture have been publicly released as Open Source Models in compliance with Section 2.2(a). AMD must respond within thirty (30) days of any such request.



SECTION 7 — DISCLAIMERS \& LIMITATIONS

7.1 No Warranty.

THE WEAVE ARCHITECTURE IS PROVIDED "AS IS," WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR NON-INFRINGEMENT. THE AUTHOR DOES NOT WARRANT THAT THE ARCHITECTURE WILL MEET ANY SPECIFIC PERFORMANCE REQUIREMENTS OR OPERATE WITHOUT ERROR.



7.2 Limitation of Liability.

IN NO EVENT SHALL THOMAS PRICE BE LIABLE FOR ANY INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES ARISING OUT OF OR IN CONNECTION WITH THE USE OR INABILITY TO USE THE WEAVE ARCHITECTURE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.



SECTION 8 — GENERAL PROVISIONS

8.1 Governing Law.

This License shall be governed by and construed in accordance with the laws of the Commonwealth of Kentucky, United States of America, without regard to its conflict of law provisions.



8.2 Severability.

If any provision of this License is held invalid or unenforceable, the remaining provisions shall continue in full force and effect.



8.3 Entire Agreement.

This License constitutes the entire agreement between the parties with respect to the subject matter herein and supersedes all prior agreements, representations, and understandings.



8.4 Modifications.

The Author reserves the right to issue revised versions of this License. Revised versions apply only to future releases; existing grants under a prior version remain governed by that version unless the grantee agrees in writing to the new terms.



8.5 Contact.

For licensing inquiries, permission requests, or compliance matters, contact:

Thomas Price — thomaspricetj@gmail.com



Weave Architecture Proprietary License — Version 1.0

Effective Date: September 18, 2026

Copyright © 2026 Thomas Price. All rights reserved.





