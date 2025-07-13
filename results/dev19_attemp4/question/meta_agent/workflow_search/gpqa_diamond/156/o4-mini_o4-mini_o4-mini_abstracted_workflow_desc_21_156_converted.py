async def forward_156(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: SC_CoT - Analyze outbreak scenario
    sc_inst_0 = (
        "Sub-task 1: Analyze the outbreak scenario, extract retrovirus characteristics, specimen types, "
        "resource constraints, and rapid-detection requirements."
    )
    N0 = self.max_sc
    sc_agents_0 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N0)
    ]
    possible_thinkings_0 = []
    possible_answers_0 = []
    subtask0 = {"subtask_id": "stage_0_subtask_1", "instruction": sc_inst_0, "agent_collaboration": "SC_CoT", "context": ["user query"]}
    for i in range(N0):
        thinking0_i, answer0_i = await sc_agents_0[i]([taskInfo], sc_inst_0, is_sub_task=True)
        agents.append(f"CoT-SC agent {sc_agents_0[i].id}, analyzing outbreak, thinking: {thinking0_i.content}; answer: {answer0_i.content}")
        possible_thinkings_0.append(thinking0_i)
        possible_answers_0.append(answer0_i)
    final_decider_0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decider_0(
        [taskInfo] + possible_thinkings_0 + possible_answers_0,
        "Sub-task 1: Synthesize and choose the most consistent analysis of the outbreak scenario.",
        is_sub_task=True
    )
    subtask0['response'] = {"thinking": thinking0, "answer": answer0}
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    logs.append(subtask0)
    print("Step 1: ", sub_tasks[-1])

    # Stage 1 Subtask 1: Debate - Generate candidate diagnostic approaches
    debate_instr = (
        "Given solutions to the problem from other agents, consider their opinions as additional advice. "
        "Please think carefully and provide an updated answer."
    )
    inst_1_1 = (
        "Sub-task 2: Generate a set of candidate diagnostic approaches: conventional PCR, nested PCR, "
        "real-time RT-PCR, ELISA-based IgG detection."
    ) + debate_instr
    debate_agents = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    R1 = self.max_round
    all_thinking_1 = [[] for _ in range(R1)]
    all_answer_1 = [[] for _ in range(R1)]
    subtask1 = {"subtask_id": "stage_1_subtask_1", "instruction": inst_1_1, "agent_collaboration": "Debate", "context": ["user query", "analysis output"]}
    for r in range(R1):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking1_i, answer1_i = await agent([taskInfo, thinking0, answer0], inst_1_1, is_sub_task=True)
            else:
                inp = [taskInfo, thinking0, answer0] + all_thinking_1[r-1] + all_answer_1[r-1]
                thinking1_i, answer1_i = await agent(inp, inst_1_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
            all_thinking_1[r].append(thinking1_i)
            all_answer_1[r].append(answer1_i)
    decider_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await decider_1(
        [taskInfo, thinking0, answer0] + all_thinking_1[-1] + all_answer_1[-1],
        "Sub-task 2: Provide final set of candidate diagnostic approaches by reasoning over debate agent outputs. "
        "Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    subtask1['response'] = {"thinking": thinking1, "answer": answer1}
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking1.content}; answer - {answer1.content}")
    logs.append(subtask1)
    print("Step 2: ", sub_tasks[-1])

    # Stage 1 Subtask 2: SC_CoT - Evaluate and prioritize candidates
    sc_inst_2 = (
        "Sub-task 3: Evaluate and prioritize each candidate diagnostic approach by sensitivity, specificity, "
        "time-to-result, equipment needs, and suitability for early infection detection."
    )
    N2 = self.max_sc
    sc_agents_2 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N2)
    ]
    possible_thinkings_2 = []
    possible_answers_2 = []
    subtask2 = {"subtask_id": "stage_1_subtask_2", "instruction": sc_inst_2, "agent_collaboration": "SC_CoT", "context": ["user query", "analysis output", "candidates output"]}
    for i in range(N2):
        thinking2_i, answer2_i = await sc_agents_2[i](
            [taskInfo, thinking0, answer0, thinking1, answer1], sc_inst_2, is_sub_task=True
        )
        agents.append(f"CoT-SC agent {sc_agents_2[i].id}, evaluating candidates, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_thinkings_2.append(thinking2_i)
        possible_answers_2.append(answer2_i)
    decider_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await decider_2(
        [taskInfo, thinking0, answer0, thinking1, answer1] + possible_thinkings_2 + possible_answers_2,
        "Sub-task 3: Synthesize and choose the most consistent evaluation and prioritization of diagnostic approaches.",
        is_sub_task=True
    )
    subtask2['response'] = {"thinking": thinking2, "answer": answer2}
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking2.content}; answer - {answer2.content}")
    logs.append(subtask2)
    print("Step 3: ", sub_tasks[-1])

    # Stage 2: CoT - Design final diagnostic kit workflow
    cot_inst_3 = (
        "Sub-task 4: Design the final diagnostic kit workflow for the top-ranked approach: specify sample preparation, "
        "RNA extraction, reverse transcription, primer/probe design, limit of detection, Ct thresholds, controls, "
        "and projected turnaround time."
    )
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask3 = {"subtask_id": "stage_2_subtask_1", "instruction": cot_inst_3, "agent_collaboration": "CoT", "context": ["user query", "analysis output", "candidates output", "evaluation output"]}
    thinking3, answer3 = await cot_agent_3(
        [taskInfo, thinking0, answer0, thinking1, answer1, thinking2, answer2], cot_inst_3, is_sub_task=True
    )
    agents.append(f"CoT agent {cot_agent_3.id}, designing workflow, thinking: {thinking3.content}; answer: {answer3.content}")
    subtask3['response'] = {"thinking": thinking3, "answer": answer3}
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking3.content}; answer - {answer3.content}")
    logs.append(subtask3)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs