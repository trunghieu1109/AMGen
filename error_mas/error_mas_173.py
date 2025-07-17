async def forward_173(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Compute rest masses m1 and m2 using SC_CoT
    N1 = self.max_sc
    cot_sc_agents1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": "Sub-task 1: Compute the rest masses m1 and m2 of the two fragments from m1 + m2 = 0.99 M and m1 = 2 m2.",
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for agent in cot_sc_agents1:
        thinking, answer = await agent([taskInfo], subtask_desc1["instruction"], is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings1.append(thinking)
        possible_answers1.append(answer)
    final_decision_agent1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    decision_inst1 = "Sub-task 1: Synthesize and choose the most consistent solutions for m1 and m2 given the above outputs."
    thinking1_final, answer1_final = await final_decision_agent1([taskInfo] + possible_thinkings1 + possible_answers1, decision_inst1, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1_final.content}; answer - {answer1_final.content}")
    subtask_desc1['response'] = {"thinking": thinking1_final, "answer": answer1_final}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 2: Formulate kinematic equations using SC_CoT
    N2 = self.max_sc
    cot_sc_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": "Sub-task 2: Formulate equations for T1 and T2 classically (T = p^2/(2m)) and relativistically (T = sqrt(p^2 + m^2) - m), imposing T1 + T2 = 0.01 M c^2 and |p1| = |p2|.",
        "context": ["user query", "Sub-task 1 thinking and answer"],
        "agent_collaboration": "SC_CoT"
    }
    for agent in cot_sc_agents2:
        thinking, answer = await agent([taskInfo, thinking1_final, answer1_final], subtask_desc2["instruction"], is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings2.append(thinking)
        possible_answers2.append(answer)
    final_decision_agent2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    decision_inst2 = "Sub-task 2: Synthesize and choose the most consistent kinematic equations given the above outputs."
    thinking2_final, answer2_final = await final_decision_agent2([taskInfo, thinking1_final, answer1_final] + possible_thinkings2 + possible_answers2, decision_inst2, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_final.content}; answer - {answer2_final.content}")
    subtask_desc2['response'] = {"thinking": thinking2_final, "answer": answer2_final}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 3: Solve for momentum and compute T1 values using SC_CoT
    N3 = self.max_sc
    cot_sc_agents3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N3)]
    possible_thinkings3 = []
    possible_answers3 = []
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": "Sub-task 3: Solve the combined equations for the common momentum p, then calculate numeric values of T1 (classical and relativistic) given M c^2 = 300 GeV.",
        "context": ["user query", "Sub-task 2 thinking and answer"],
        "agent_collaboration": "SC_CoT"
    }
    for agent in cot_sc_agents3:
        thinking, answer = await agent([taskInfo, thinking2_final, answer2_final], subtask_desc3["instruction"], is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings3.append(thinking)
        possible_answers3.append(answer)
    final_decision_agent3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    decision_inst3 = "Sub-task 3: Synthesize and choose the most consistent numeric T1_classical and T1_relativistic values."
    thinking3_final, answer3_final = await final_decision_agent3([taskInfo, thinking2_final, answer2_final] + possible_thinkings3 + possible_answers3, decision_inst3, is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3_final.content}; answer - {answer3_final.content}")
    subtask_desc3['response'] = {"thinking": thinking3_final, "answer": answer3_final}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 4: Compute Δ and select closest choice using Debate
    N4 = self.max_round
    debate_agents4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking4 = [[] for _ in range(N4)]
    all_answer4 = [[] for _ in range(N4)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": "Sub-task 4: Compute Δ = T1_rel - T1_classical (in MeV) and select the closest value from {2 MeV, 5 MeV, 10 MeV, 20 MeV}. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.",
        "context": ["user query", "Sub-task 3 thinking and answer"],
        "agent_collaboration": "Debate"
    }
    for r in range(N4):
        for agent in debate_agents4:
            if r == 0:
                thinking, answer = await agent([taskInfo, thinking3_final, answer3_final], subtask_desc4["instruction"], r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking3_final, answer3_final] + all_thinking4[r-1] + all_answer4[r-1]
                thinking, answer = await agent(inputs, subtask_desc4["instruction"], r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking4[r].append(thinking)
            all_answer4[r].append(answer)
    final_decision_agent4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_inst4 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking4_final, answer4_final = await final_decision_agent4([taskInfo, thinking3_final, answer3_final] + all_thinking4[-1] + all_answer4[-1], final_inst4, is_sub_task=True)
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4_final.content}; answer - {answer4_final.content}")
    subtask_desc4['response'] = {"thinking": thinking4_final, "answer": answer4_final}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer, logs = await self.make_final_answer(thinking4_final, answer4_final, sub_tasks, agents)
    return final_answer, logs