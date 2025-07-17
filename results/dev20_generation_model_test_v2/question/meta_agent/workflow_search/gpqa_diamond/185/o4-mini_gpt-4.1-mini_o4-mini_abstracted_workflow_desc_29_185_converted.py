async def forward_185(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: Extract and summarize structural and reaction information (SC_CoT)
    cot_sc_inst0 = (
        "Sub-task 0_1: Analyze the substrate and reaction information, "
        "extracting structural details including substrate identity, stereochemistry, reaction type, and the four product choices."
    )
    N0 = self.max_sc
    cot_sc_agents0 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N0)
    ]
    possible_thinkings0 = []
    possible_answers0 = []
    subtask_desc0_1 = {
        "subtask_id": "subtask_0_1",
        "instruction": cot_sc_inst0,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N0):
        thinking0, answer0 = await cot_sc_agents0[i]([taskInfo], cot_sc_inst0, is_sub_task=True)
        agents.append(
            f"CoT-SC agent {cot_sc_agents0[i].id}, analyzing substrate and reaction, thinking: {thinking0.content}; answer: {answer0.content}"
        )
        possible_thinkings0.append(thinking0)
        possible_answers0.append(answer0)
    final_decision_agent0 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0_final, answer0_final = await final_decision_agent0(
        [taskInfo] + possible_thinkings0 + possible_answers0,
        "Sub-task 0_1: Synthesize the most consistent extraction of structural and reaction information.",
        is_sub_task=True
    )
    sub_tasks.append(
        f"Sub-task 0_1 output: thinking - {thinking0_final.content}; answer - {answer0_final.content}"
    )
    subtask_desc0_1['response'] = {"thinking": thinking0_final, "answer": answer0_final}
    logs.append(subtask_desc0_1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 1: Derive rearranged skeleton via debate (Debate)
    debate_instr1 = (
        "Sub-task 1_1: Apply the concerted [3,3]-sigmatropic mechanism to the bicyclic substrate "
        "to derive the rearranged cyclopentapyridine skeleton, tracking bond shifts and stereochemistry. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. "
        "Please think carefully and provide an updated answer."
    )
    debate_agents1 = [
        LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
        for role in self.debate_role
    ]
    all_thinking1 = [[] for _ in range(self.max_round)]
    all_answer1 = [[] for _ in range(self.max_round)]
    subtask_desc1_1 = {
        "subtask_id": "subtask_1_1",
        "instruction": debate_instr1,
        "context": ["user query", "response of subtask_0_1"],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        for agent in debate_agents1:
            if r == 0:
                thinking1, answer1 = await agent(
                    [taskInfo, thinking0_final, answer0_final], debate_instr1, r, is_sub_task=True
                )
            else:
                thinking1, answer1 = await agent(
                    [taskInfo, thinking0_final, answer0_final] + all_thinking1[r-1] + all_answer1[r-1],
                    debate_instr1, r, is_sub_task=True
                )
            agents.append(
                f"Debate agent {agent.id}, round {r}, deriving rearranged skeleton, thinking: {thinking1.content}; answer: {answer1.content}"
            )
            all_thinking1[r].append(thinking1)
            all_answer1[r].append(answer1)
    final_decision_agent1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_final, answer1_final = await final_decision_agent1(
        [taskInfo, thinking0_final, answer0_final] + all_thinking1[-1] + all_answer1[-1],
        "Sub-task 1_1: Given all the above thinking and answers, reason over them carefully and provide a final arrangement of the cyclopentapyridine skeleton.",
        is_sub_task=True
    )
    agents.append(
        f"Final Decision agent, synthesizing rearranged skeleton, thinking: {thinking1_final.content}; answer: {answer1_final.content}"
    )
    sub_tasks.append(
        f"Sub-task 1_1 output: thinking - {thinking1_final.content}; answer - {answer1_final.content}"
    )
    subtask_desc1_1['response'] = {"thinking": thinking1_final, "answer": answer1_final}
    logs.append(subtask_desc1_1)
    print("Step 2: ", sub_tasks[-1])

    # Stage 2: Map skeleton to product choices (SC_CoT)
    cot_sc_inst2 = (
        "Sub-task 2_1: Map the rearranged skeleton onto each of the four cyclopenta[c]pyridine nomenclatures "
        "to determine which patterns of unsaturation and hydrogenation match."
    )
    N2 = self.max_sc
    cot_sc_agents2 = [
        LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
        for _ in range(N2)
    ]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2_1 = {
        "subtask_id": "subtask_2_1",
        "instruction": cot_sc_inst2,
        "context": ["user query", "response of subtask_1_1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N2):
        thinking2, answer2 = await cot_sc_agents2[i](
            [taskInfo, thinking1_final, answer1_final], cot_sc_inst2, is_sub_task=True
        )
        agents.append(
            f"CoT-SC agent {cot_sc_agents2[i].id}, mapping skeleton to nomenclatures, thinking: {thinking2.content}; answer: {answer2.content}"
        )
        possible_thinkings2.append(thinking2)
        possible_answers2.append(answer2)
    final_decision_agent2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2_final, answer2_final = await final_decision_agent2(
        [taskInfo, thinking1_final, answer1_final] + possible_thinkings2 + possible_answers2,
        "Sub-task 2_1: Synthesize and choose the nomenclature that matches the rearranged skeleton.",
        is_sub_task=True
    )
    sub_tasks.append(
        f"Sub-task 2_1 output: thinking - {thinking2_final.content}; answer - {answer2_final.content}"
    )
    subtask_desc2_1['response'] = {"thinking": thinking2_final, "answer": answer2_final}
    logs.append(subtask_desc2_1)
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking2_final, answer2_final, sub_tasks, agents)
    return final_answer, logs