async def forward_155(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr_1 = (
        "Sub-task 1: Perform a rigorous stereochemical analysis of the epoxide products formed from the epoxidation of (E)-oct-4-ene and (Z)-oct-4-ene with one equivalent of mCPBA followed by aqueous acid. "
        "Explicitly determine the number and stereochemical nature of the stereoisomers formed by: (a) checking for molecular symmetry or meso forms, (b) assessing whether each reaction produces racemic mixtures or single enantiomers due to the stereospecific syn addition mechanism of mCPBA, and (c) considering the effect of aqueous acid workup on stereochemistry. "
        "Avoid assuming all products are racemic pairs. This subtask addresses the root cause of previous errors by embedding explicit meso and stereospecificity analysis. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )

    debate_agents_1 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]

    all_thinking_1 = []
    all_answer_1 = []

    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instr_1,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }

    for i, agent in enumerate(debate_agents_1):
        thinking1, answer1 = await agent([taskInfo], debate_instr_1, 0, is_sub_task=True)
        agents.append(f"Debate agent {agent.id}, round 0, stereochemical analysis, thinking: {thinking1.content}; answer: {answer1.content}")
        all_thinking_1.append(thinking1)
        all_answer_1.append(answer1)

    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1(
        [taskInfo] + all_thinking_1 + all_answer_1,
        "Sub-task 1: Synthesize and choose the most consistent stereochemical analysis." +
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.",
        is_sub_task=True
    )

    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction_2 = (
        "Sub-task 2: Based on the stereochemical analysis from Sub-task 1, analyze the chromatographic behavior of the combined epoxide mixture on a standard (achiral) reverse-phase HPLC column. "
        "Determine which stereoisomers co-elute or separate, considering that achiral columns do not resolve enantiomers but can separate diastereomers. "
        "Incorporate the assumption of ideal chromatographic resolution and explicitly link chromatographic behavior to the stereochemical nature (meso, enantiomer, diastereomer) of the products."
    )

    N_sc = self.max_sc
    cot_agents_2 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N_sc)
    ]

    possible_answers_2 = []
    possible_thinkings_2 = []

    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }

    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1, answer1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, analyze achiral HPLC behavior, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)

    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2(
        [taskInfo, thinking1, answer1] + possible_thinkings_2 + possible_answers_2,
        "Sub-task 2: Synthesize and choose the most consistent chromatographic behavior on achiral HPLC.",
        is_sub_task=True
    )

    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    cot_sc_instruction_3 = (
        "Sub-task 3: Based on the stereochemical analysis from Sub-task 1, analyze the chromatographic behavior of the combined epoxide mixture on a chiral HPLC column. "
        "Determine how enantiomers and diastereomers are resolved, considering that chiral columns can separate enantiomers as well as diastereomers. "
        "Use the ideal chromatographic resolution assumption and explicitly consider the presence or absence of enantiomeric pairs from each epoxide product."
    )

    cot_agents_3 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N_sc)
    ]

    possible_answers_3 = []
    possible_thinkings_3 = []

    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction_3,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }

    for i in range(N_sc):
        thinking3, answer3 = await cot_agents_3[i]([taskInfo, thinking1, answer1], cot_sc_instruction_3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_3[i].id}, analyze chiral HPLC behavior, thinking: {thinking3.content}; answer: {answer3.content}")
        possible_answers_3.append(answer3)
        possible_thinkings_3.append(thinking3)

    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3(
        [taskInfo, thinking1, answer1] + possible_thinkings_3 + possible_answers_3,
        "Sub-task 3: Synthesize and choose the most consistent chromatographic behavior on chiral HPLC.",
        is_sub_task=True
    )

    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    debate_instr_4 = (
        "Sub-task 4: Integrate the stereochemical and chromatographic analyses from Subtasks 1, 2, and 3 to predict the total number of peaks observed in both the standard (achiral) and chiral HPLC chromatograms. "
        "Use this integrated understanding to select the correct answer choice. This subtask must explicitly avoid previous errors by relying on the corrected stereochemical assumptions and chromatographic behavior, ensuring a logically consistent final conclusion. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )

    debate_agents_4 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]

    all_thinking_4 = []
    all_answer_4 = []

    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instr_4,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1", "thinking of subtask 2", "answer of subtask 2", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Debate"
    }

    for i, agent in enumerate(debate_agents_4):
        thinking4, answer4 = await agent(
            [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3],
            debate_instr_4, 0, is_sub_task=True
        )
        agents.append(f"Debate agent {agent.id}, round 0, integration and final prediction, thinking: {thinking4.content}; answer: {answer4.content}")
        all_thinking_4.append(thinking4)
        all_answer_4.append(answer4)

    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent_4(
        [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3] + all_thinking_4 + all_answer_4,
        "Sub-task 4: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )

    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs
