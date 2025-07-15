async def forward_165(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction_0 = (
        "Sub-task 1: Extract and summarize all given quantitative and qualitative information from the Lagrangian, field content, "
        "vacuum expectation values (VEVs), and problem statement relevant for the pseudo-Goldstone boson mass calculation. "
        "Include clarifications on particle content, gauge quantum numbers, VEV definitions, and identify all particles contributing to radiative corrections. "
        "Embed the feedback that incomplete or superficial extraction can lead to missing key contributions such as the top quark loop."
    )
    N_sc_0 = self.max_sc
    cot_agents_0 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_0)]
    possible_answers_0 = []
    possible_thinkings_0 = []
    subtask_desc_0 = {
        "subtask_id": "stage0_subtask1",
        "instruction": cot_sc_instruction_0,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_0):
        thinking0, answer0 = await cot_agents_0[i]([taskInfo], cot_sc_instruction_0, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0[i].id}, extracting and summarizing info, thinking: {thinking0.content}; answer: {answer0.content}")
        possible_answers_0.append(answer0)
        possible_thinkings_0.append(thinking0)
    final_decision_agent_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent_0(
        [taskInfo] + possible_thinkings_0 + possible_answers_0,
        "Sub-task 1: Synthesize and choose the most consistent and correct extraction and summary of given information for the pseudo-Goldstone boson mass calculation.",
        is_sub_task=True
    )
    sub_tasks.append(f"Stage 0 Sub-task 1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc_0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc_0)
    print("Step 0: ", sub_tasks[-1])

    debate_instruction_1_1 = (
        "Sub-task 1: Compute the one-loop effective potential contributions explicitly for each relevant particle species (scalars, gauge bosons, fermions) in the model, "
        "listing their contributions with correct signs and multiplicities. Pay special attention to the top quark contribution, verifying it is included as the dominant negative fermion loop term. "
        "Given the extraction from Stage 0, ensure no contributions are omitted."
    )
    debate_agents_1_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_1_1 = self.max_round
    all_thinking_1_1 = [[] for _ in range(N_max_1_1)]
    all_answer_1_1 = [[] for _ in range(N_max_1_1)]
    subtask_desc_1_1 = {
        "subtask_id": "stage1_subtask1",
        "instruction": debate_instruction_1_1,
        "context": ["user query", thinking0, answer0],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1_1):
        for i, agent in enumerate(debate_agents_1_1):
            if r == 0:
                thinking1_1, answer1_1 = await agent([taskInfo, thinking0, answer0], debate_instruction_1_1, r, is_sub_task=True)
            else:
                input_infos_1_1 = [taskInfo, thinking0, answer0] + all_thinking_1_1[r-1] + all_answer_1_1[r-1]
                thinking1_1, answer1_1 = await agent(input_infos_1_1, debate_instruction_1_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, computing one-loop contributions, thinking: {thinking1_1.content}; answer: {answer1_1.content}")
            all_thinking_1_1[r].append(thinking1_1)
            all_answer_1_1[r].append(answer1_1)
    final_decision_agent_1_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_1, answer1_1 = await final_decision_agent_1_1(
        [taskInfo, thinking0, answer0] + all_thinking_1_1[-1] + all_answer_1_1[-1],
        "Sub-task 1: Synthesize and finalize the one-loop effective potential contributions for all relevant particles.",
        is_sub_task=True
    )
    sub_tasks.append(f"Stage 1 Sub-task 1 output: thinking - {thinking1_1.content}; answer - {answer1_1.content}")
    subtask_desc_1_1['response'] = {"thinking": thinking1_1, "answer": answer1_1}
    logs.append(subtask_desc_1_1)
    print("Step 1.1: ", sub_tasks[-1])

    cot_sc_instruction_1_2 = (
        "Sub-task 2: Assemble the approximate formula for the pseudo-Goldstone boson mass squared, M_h2^2, by taking the second derivative of the full Coleman-Weinberg effective potential derived in Sub-task 1. "
        "Explicitly include all terms, verify the normalization factor placement, and confirm the presence and sign of the top quark mass term. "
        "Enforce a consistency check that the top Yukawa coupling dominates the radiative corrections."
    )
    N_sc_1_2 = self.max_sc
    cot_agents_1_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc_1_2)]
    possible_answers_1_2 = []
    possible_thinkings_1_2 = []
    subtask_desc_1_2 = {
        "subtask_id": "stage1_subtask2",
        "instruction": cot_sc_instruction_1_2,
        "context": ["user query", thinking0, answer0, thinking1_1, answer1_1],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_1_2):
        thinking1_2, answer1_2 = await cot_agents_1_2[i]([taskInfo, thinking0, answer0, thinking1_1, answer1_1], cot_sc_instruction_1_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1_2[i].id}, assembling mass formula, thinking: {thinking1_2.content}; answer: {answer1_2.content}")
        possible_answers_1_2.append(answer1_2)
        possible_thinkings_1_2.append(thinking1_2)
    final_decision_agent_1_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_2, answer1_2 = await final_decision_agent_1_2(
        [taskInfo, thinking0, answer0, thinking1_1, answer1_1] + possible_thinkings_1_2 + possible_answers_1_2,
        "Sub-task 2: Synthesize and finalize the approximate formula for the pseudo-Goldstone boson mass squared, ensuring correctness and consistency.",
        is_sub_task=True
    )
    sub_tasks.append(f"Stage 1 Sub-task 2 output: thinking - {thinking1_2.content}; answer - {answer1_2.content}")
    subtask_desc_1_2['response'] = {"thinking": thinking1_2, "answer": answer1_2}
    logs.append(subtask_desc_1_2)
    print("Step 1.2: ", sub_tasks[-1])

    debate_instruction_2_1 = (
        "Sub-task 1: Evaluate the four candidate mass formulas by comparing their structure, terms, signs, and normalization against the derived formula from Stage 1 Sub-task 2 and known theoretical expectations. "
        "Critically assess the inclusion and sign of the top quark term and other particle contributions. Use a Debate pattern to challenge assumptions and avoid premature consensus. "
        "Ensure the final choice is physically and mathematically justified."
    )
    debate_agents_2_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max_2_1 = self.max_round
    all_thinking_2_1 = [[] for _ in range(N_max_2_1)]
    all_answer_2_1 = [[] for _ in range(N_max_2_1)]
    subtask_desc_2_1 = {
        "subtask_id": "stage2_subtask1",
        "instruction": debate_instruction_2_1,
        "context": ["user query", thinking0, answer0, thinking1_2, answer1_2],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2_1):
        for i, agent in enumerate(debate_agents_2_1):
            if r == 0:
                thinking2_1, answer2_1 = await agent([taskInfo, thinking0, answer0, thinking1_2, answer1_2], debate_instruction_2_1, r, is_sub_task=True)
            else:
                input_infos_2_1 = [taskInfo, thinking0, answer0, thinking1_2, answer1_2] + all_thinking_2_1[r-1] + all_answer_2_1[r-1]
                thinking2_1, answer2_1 = await agent(input_infos_2_1, debate_instruction_2_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, evaluating candidate formulas, thinking: {thinking2_1.content}; answer: {answer2_1.content}")
            all_thinking_2_1[r].append(thinking2_1)
            all_answer_2_1[r].append(answer2_1)
    final_decision_agent_2_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2_1, answer2_1 = await final_decision_agent_2_1(
        [taskInfo, thinking0, answer0, thinking1_2, answer1_2] + all_thinking_2_1[-1] + all_answer_2_1[-1],
        "Sub-task 1: Synthesize and select the most physically and mathematically justified candidate formula for the pseudo-Goldstone boson mass.",
        is_sub_task=True
    )
    sub_tasks.append(f"Stage 2 Sub-task 1 output: thinking - {thinking2_1.content}; answer - {answer2_1.content}")
    subtask_desc_2_1['response'] = {"thinking": thinking2_1, "answer": answer2_1}
    logs.append(subtask_desc_2_1)
    print("Step 2.1: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking2_1, answer2_1, sub_tasks, agents)
    return final_answer, logs
