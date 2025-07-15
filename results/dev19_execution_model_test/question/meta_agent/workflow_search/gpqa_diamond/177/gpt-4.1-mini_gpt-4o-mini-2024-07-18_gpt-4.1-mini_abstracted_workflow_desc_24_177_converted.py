async def forward_177(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr_stage0 = (
        "Sub-task 1: Determine the canonical mass dimensions of the fields psi, F^{mu nu}, and the operator sigma_{mu nu}, "
        "and use these to find the mass dimension of the coupling constant kappa in the interaction Lagrangian. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_stage0 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_stage0 = self.max_round
    all_thinking_stage0 = [[] for _ in range(N_max_stage0)]
    all_answer_stage0 = [[] for _ in range(N_max_stage0)]
    subtask_desc0 = {
        "subtask_id": "stage_0.subtask_1",
        "instruction": debate_instr_stage0,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_stage0):
        for i, agent in enumerate(debate_agents_stage0):
            if r == 0:
                thinking0, answer0 = await agent([taskInfo], debate_instr_stage0, r, is_sub_task=True)
            else:
                input_infos_0 = [taskInfo] + all_thinking_stage0[r-1] + all_answer_stage0[r-1]
                thinking0, answer0 = await agent(input_infos_0, debate_instr_stage0, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking0.content}; answer: {answer0.content}")
            all_thinking_stage0[r].append(thinking0)
            all_answer_stage0[r].append(answer0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent_0(
        [taskInfo] + all_thinking_stage0[-1] + all_answer_stage0[-1],
        "Sub-task 1: Synthesize and choose the most consistent and correct mass dimension of kappa.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent stage 0, thinking: {thinking0.content}; answer: {answer0.content}")
    sub_tasks.append(f"Stage 0 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc0)
    print("Step 0: ", sub_tasks[-1])

    cot_sc_instruction_stage1 = (
        "Sub-task 1: Based on the output from Stage 0, analyze how the mass dimension of kappa affects the renormalizability of the theory, "
        "using standard QFT criteria for operator dimensions and coupling constants."
    )
    N_sc = self.max_sc
    cot_agents_stage1 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N_sc)
    ]
    possible_answers_stage1 = []
    possible_thinkings_stage1 = []
    subtask_desc1 = {
        "subtask_id": "stage_1.subtask_1",
        "instruction": cot_sc_instruction_stage1,
        "context": ["user query", thinking0.content, answer0.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking1, answer1 = await cot_agents_stage1[i]([taskInfo, thinking0, answer0], cot_sc_instruction_stage1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_stage1[i].id}, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers_stage1.append(answer1)
        possible_thinkings_stage1.append(thinking1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1(
        [taskInfo, thinking0, answer0] + possible_thinkings_stage1 + possible_answers_stage1,
        "Sub-task 1: Synthesize and choose the most consistent and correct renormalizability conclusion.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent stage 1, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Stage 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    debate_instr_stage2 = (
        "Sub-task 1: Evaluate the four given answer choices by matching the calculated mass dimension and renormalizability conclusion to select the correct option. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_stage2 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    N_max_stage2 = self.max_round
    all_thinking_stage2 = [[] for _ in range(N_max_stage2)]
    all_answer_stage2 = [[] for _ in range(N_max_stage2)]
    subtask_desc2 = {
        "subtask_id": "stage_2.subtask_1",
        "instruction": debate_instr_stage2,
        "context": ["user query", thinking0.content, answer0.content, thinking1.content, answer1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_stage2):
        for i, agent in enumerate(debate_agents_stage2):
            if r == 0:
                thinking2, answer2 = await agent([taskInfo, thinking0, answer0, thinking1, answer1], debate_instr_stage2, r, is_sub_task=True)
            else:
                input_infos_2 = [taskInfo, thinking0, answer0, thinking1, answer1] + all_thinking_stage2[r-1] + all_answer_stage2[r-1]
                thinking2, answer2 = await agent(input_infos_2, debate_instr_stage2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking_stage2[r].append(thinking2)
            all_answer_stage2[r].append(answer2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent_2(
        [taskInfo, thinking0, answer0, thinking1, answer1] + all_thinking_stage2[-1] + all_answer_stage2[-1],
        "Sub-task 1: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    agents.append(f"Final Decision agent stage 2, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Stage 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking2, answer2, sub_tasks, agents)
    return final_answer, logs
