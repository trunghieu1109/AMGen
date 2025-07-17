async def forward_172(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Sub-task 1: Extract and convert input data using SC-CoT
    cot_sc_instruction1 = (
        "Sub-task 1: Extract the given data and convert units: electron speed v, "
        "position uncertainty Δx, and identify constants ħ and m_e."
    )
    N1 = self.max_sc
    cot_agents1 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction1,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N1):
        thinking1_i, answer1_i = await cot_agents1[i]([taskInfo], cot_sc_instruction1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents1[i].id}, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
        possible_thinkings1.append(thinking1_i)
        possible_answers1.append(answer1_i)
    final_decision_agent1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent1(
        [taskInfo] + possible_thinkings1 + possible_answers1,
        "Sub-task 1: Synthesize and choose the most consistent answer for extracting data.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    agents.append(f"Final Decision Agent {final_decision_agent1.id}, thinking: {thinking1.content}; answer: {answer1.content}")
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Compute Δp using Δx·Δp ≥ ħ/2 via SC-CoT
    cot_sc_instruction2 = (
        "Sub-task 2: Apply Heisenberg uncertainty Δx·Δp ≥ ħ/2 to compute minimum momentum uncertainty Δp."
    )
    N2 = self.max_sc
    cot_agents2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N2)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction2,
        "context": ["user query", "thinking of subtask 1", "answer of subtask 1"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N2):
        thinking2_i, answer2_i = await cot_agents2[i]([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents2[i].id}, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_thinkings2.append(thinking2_i)
        possible_answers2.append(answer2_i)
    final_decision_agent2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision_agent2(
        [taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2,
        "Sub-task 2: Synthesize and choose the most consistent answer for computing Δp.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    agents.append(f"Final Decision Agent {final_decision_agent2.id}, thinking: {thinking2.content}; answer: {answer2.content}")
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Estimate ΔE via ΔE ≈ v·Δp using SC-CoT
    cot_sc_instruction3 = (
        "Sub-task 3: Using Δp from Sub-task 2 and velocity v, estimate the minimum energy uncertainty ΔE via ΔE ≈ v·Δp."
    )
    N3 = self.max_sc
    cot_agents3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N3)]
    possible_thinkings3 = []
    possible_answers3 = []
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_sc_instruction3,
        "context": ["user query", "thinking of subtask 2", "answer of subtask 2"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N3):
        thinking3_i, answer3_i = await cot_agents3[i]([taskInfo, thinking2, answer2], cot_sc_instruction3, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents3[i].id}, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
        possible_thinkings3.append(thinking3_i)
        possible_answers3.append(answer3_i)
    final_decision_agent3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent3(
        [taskInfo, thinking2, answer2] + possible_thinkings3 + possible_answers3,
        "Sub-task 3: Synthesize and choose the most consistent answer for estimating ΔE.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    agents.append(f"Final Decision Agent {final_decision_agent3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Compare ΔE to options via Debate
    debate_instr4 = (
        "Sub-task 4: Compare the computed ΔE with given multiple-choice options and select the closest match."
        " Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking4 = [[] for _ in range(self.max_round)]
    all_answer4 = [[] for _ in range(self.max_round)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instr4,
        "context": ["user query", "thinking of subtask 3", "answer of subtask 3"],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents4):
            if r == 0:
                thinking4_i, answer4_i = await agent([taskInfo, thinking3, answer3], debate_instr4, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4_i, answer4_i = await agent(inputs, debate_instr4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
            all_thinking4[r].append(thinking4_i)
            all_answer4[r].append(answer4_i)
    final_decision_agent4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking4, answer4 = await final_decision_agent4(
        [taskInfo, thinking3, answer3] + all_thinking4[-1] + all_answer4[-1],
        "Sub-task 4: Given all the above thinking and answers, reason over them carefully and provide a final answer.",
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    agents.append(f"Final Decision Agent {final_decision_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs