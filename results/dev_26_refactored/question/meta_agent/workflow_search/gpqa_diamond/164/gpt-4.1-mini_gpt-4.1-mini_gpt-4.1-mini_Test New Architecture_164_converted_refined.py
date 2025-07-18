async def forward_164(self, taskInfo):
    print("Task Requirement: ", taskInfo)
    logs = []

    cot_instruction1 = (
        "Sub-task 1: Extract and structure all relevant information from the query, including the polymerization setup, catalyst systems, and the four statements. "
        "Ensure clarity on the 'only ethylene' monomer condition and the nature of the dual catalyst system. Avoid assumptions about industrial implementation without verification. "
        "This subtask sets a solid factual foundation for subsequent analysis."
    )
    cot_agent_desc1 = {
        "instruction": cot_instruction1,
        "input": [taskInfo],
        "temperature": 0.0,
        "context": ["user query"]
    }
    results1, log1 = await self.sc_cot(
        subtask_id="stage_1.subtask_1",
        cot_agent_desc=cot_agent_desc1,
        n_repeat=self.max_sc
    )
    logs.append(log1)

    cot_sc_instruction2 = (
        "Sub-task 2: Integrate the extracted information with authoritative chemical and industrial knowledge on ethylene polymerization, catalyst types (including group VIa transition metals and noble metals), activator compatibility, and industrial practices. "
        "Explicitly clarify ambiguous terms such as 'group VIa' metals and the 'essential additional reaction step.' This integration must carefully consider the 'only ethylene' feedstock constraint to avoid conflating with processes using comonomers or heterogeneous catalysts, addressing the previous error of unverified assumptions."
    )
    final_decision_instruction2 = (
        "Sub-task 2: Synthesize and choose the most consistent integrated knowledge for the polymerization system and catalyst statements."
    )
    cot_sc_desc2 = {
        "instruction": cot_sc_instruction2,
        "final_decision_instruction": final_decision_instruction2,
        "input": [taskInfo, results1['thinking'], results1['answer']],
        "temperature": 0.5,
        "context": ["user query", "thinking of stage_1.subtask_1", "answer of stage_1.subtask_1"]
    }
    results2, log2 = await self.sc_cot(
        subtask_id="stage_1.subtask_2",
        cot_agent_desc=cot_sc_desc2,
        n_repeat=self.max_sc
    )
    logs.append(log2)

    debate_instruction3 = (
        "Sub-task 1: Perform a dedicated Verification of each of the four statements against the integrated chemical and industrial knowledge base, "
        "with explicit cross-referencing to known industrial processes and catalyst mechanisms under the 'only ethylene' condition. "
        "This step must rigorously validate or refute each statement, avoiding acceptance of unverified premises. Use Debate collaboration to allow critical evaluation and resolution of conflicting interpretations."
    )
    final_decision_instruction3 = (
        "Sub-task 1: Verify and conclude the correctness of each of the four statements regarding the dual catalyst system for branched polyethylene formation."
    )
    debate_desc3 = {
        "instruction": debate_instruction3,
        "final_decision_instruction": final_decision_instruction3,
        "input": [taskInfo, results2['thinking'], results2['answer']],
        "context": ["user query", "thinking of stage_1.subtask_2", "answer of stage_1.subtask_2"],
        "temperature": 0.5
    }
    results3, log3 = await self.debate(
        subtask_id="stage_2.subtask_1",
        debate_desc=debate_desc3,
        n_repeat=self.max_round
    )
    logs.append(log3)

    debate_instruction4 = (
        "Sub-task 2: Conduct a Final Arbitration to reconcile any conflicting conclusions from the Verification subtask. "
        "Select the single most correct statement with clear, well-justified reasoning that prioritizes the question's core requirement—industrial feasibility of forming branched polyethylene using only ethylene and a dual catalyst system. "
        "Discourage selection based on partial truths or economic considerations alone."
    )
    final_decision_instruction4 = (
        "Sub-task 2: Select the single most correct statement regarding the dual catalyst system for branched polyethylene formation using only ethylene."
    )
    debate_desc4 = {
        "instruction": debate_instruction4,
        "final_decision_instruction": final_decision_instruction4,
        "input": [taskInfo, results3['thinking'], results3['answer']],
        "context": ["user query", "thinking of stage_2.subtask_1", "answer of stage_2.subtask_1"],
        "temperature": 0.5
    }
    results4, log4 = await self.debate(
        subtask_id="stage_2.subtask_2",
        debate_desc=debate_desc4,
        n_repeat=self.max_round
    )
    logs.append(log4)

    final_answer = await self.make_final_answer(results4['thinking'], results4['answer'])
    return final_answer, logs
