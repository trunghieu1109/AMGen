async def forward_188(self, taskInfo):
    logs = []

    cot_sc_instruction1 = (
        "Sub-task 1: Extract and summarize the physical nature and origin of each effective particle (Magnon, Skyrmion, Pion, Phonon), "
        "with emphasis on their relationship to spontaneous symmetry breaking. Explicitly analyze the physical origin of the crystal lattice formation as a spontaneous breaking of continuous translational symmetry, "
        "distinguishing spontaneous vs explicit symmetry breaking mechanisms for each particle, embedding the correction that phonons arise as Goldstone bosons from spontaneous translational symmetry breaking."
    )
    cot_sc_desc1 = {
        "instruction": cot_sc_instruction1,
        "final_decision_instruction": "Sub-task 1: Synthesize and choose the most consistent summary of the physical origin and symmetry breaking nature of each particle.",
        "input": [taskInfo],
        "temperature": 0.5,
        "context_desc": ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_sc_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_reflect_instruction2 = (
        "Sub-task 2: Perform a dedicated consistency check and validation of the classification of each particle’s symmetry breaking mechanism, "
        "especially focusing on phonons. Cite authoritative physics definitions and sources to verify the distinction between spontaneous and explicit symmetry breaking, "
        "ensuring the spontaneous nature of crystal formation and phonons as Goldstone bosons is correctly understood and accepted."
    )
    critic_instruction2 = (
        "Please review and provide the limitations or potential errors in the classification of the symmetry breaking mechanisms of Magnon, Skyrmion, Pion, and Phonon, "
        "with special attention to phonons and translational symmetry breaking."
    )
    cot_reflect_desc2 = {
        "instruction": cot_reflect_instruction2,
        "critic_instruction": critic_instruction2,
        "input": [taskInfo, results1["thinking"], results1["answer"]],
        "temperature": 0.0,
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results2, log2 = await self.reflexion(
        subtask_id="stage_1.subtask_2",
        reflect_desc=cot_reflect_desc2,
        n_repeat=self.max_round
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 3: Integrate and combine the summarized information from stage_1.subtask_1 and the consistency check results from stage_1.subtask_2 to analyze the association of each particle with spontaneously-broken symmetries. "
        "Highlight which particles are Goldstone bosons or topological excitations arising from spontaneous symmetry breaking, explicitly correcting the previous misconception about phonons and translational symmetry breaking."
    )
    final_decision_instruction3 = (
        "Sub-task 3: Provide a coherent integrated analysis of the symmetry breaking nature of Magnon, Skyrmion, Pion, and Phonon, "
        "correcting misconceptions and clarifying their physical origins."
    )
    debate_desc3 = {
        "instruction": debate_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results1["thinking"], results2["thinking"]],
        "context_desc": ["user query", "thinking of stage_1.subtask_1", "thinking of stage_1.subtask_2"],
        "temperature": 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 4: Based on the integrated and validated analysis from stage_2.subtask_1, select the effective particle that is not associated with a spontaneously-broken symmetry. "
        "Ground this final decision on the corrected conceptual framework established in previous subtasks, ensuring no prior misconceptions influence the outcome."
    )
    final_decision_instruction4 = "Sub-task 4: Select the effective particle not associated with spontaneously-broken symmetry."
    debate_desc4 = {
        "instruction": debate_instruction4,
        "final_decision_instruction": final_decision_instruction4,
        "input": [taskInfo, results3["thinking"], results3["answer"]],
        "context_desc": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"],
        "temperature": 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="stage_3.subtask_1",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4["thinking"], results4["answer"])
    return final_answer, logs
