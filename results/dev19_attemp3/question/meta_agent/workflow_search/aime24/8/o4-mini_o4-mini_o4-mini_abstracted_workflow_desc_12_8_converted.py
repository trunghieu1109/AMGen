async def forward_8(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Compute DP classification with SC-CoT
    cot_sc_instruction = (
        "Sub-task 1: Using dynamic programming, classify each n from 1 to 2024 as an N-position or P-position "
        "for the first player given removal options of 1 or 4 tokens. Describe your DP approach and result."  
    )
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5)
                  for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction,
        "context": ["user query"],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N):
        thinking1, answer1 = await cot_agents[i]([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_thinkings.append(thinking1)
        possible_answers.append(answer1)
    final_instr1 = (
        "Sub-task 1 Final: Given all the above thinking and answers, find the most consistent and correct DP classification result."  
    )
    final_agent1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1_final, answer1_final = await final_agent1(
        [taskInfo] + possible_thinkings + possible_answers,
        final_instr1,
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1_final.content}; answer - {answer1_final.content}")
    subtask_desc1['response'] = {"thinking": thinking1_final, "answer": answer1_final}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Stage 2: Derive periodic pattern with Debate
    debate_instruction_2 = (
        "Sub-task 2: Analyze the DP classification from Sub-task 1 to derive a closed-form or periodic pattern "
        "describing exactly which n are P-positions. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    )
    debate_agents_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5)
                       for role in self.debate_role]
    all_thinkings2 = []
    all_answers2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": debate_instruction_2,
        "context": ["user query", thinking1_final, answer1_final],
        "agent_collaboration": "Debate"
    }
    for r in range(self.max_round):
        all_thinkings2.append([])
        all_answers2.append([])
        for i, agent in enumerate(debate_agents_2):
            if r == 0:
                thinking2, answer2 = await agent(
                    [taskInfo, thinking1_final, answer1_final],
                    debate_instruction_2,
                    r,
                    is_sub_task=True
                )
            else:
                thinking2, answer2 = await agent(
                    [taskInfo, thinking1_final, answer1_final] + all_thinkings2[r-1] + all_answers2[r-1],
                    debate_instruction_2,
                    r,
                    is_sub_task=True
                )
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinkings2[r].append(thinking2)
            all_answers2[r].append(answer2)
    final_instr2 = (
        "Sub-task 2 Final: Given all the above thinking and answers, reason over them carefully and provide a final answer."
    )
    final_agent2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2_final, answer2_final = await final_agent2(
        [taskInfo, thinking1_final, answer1_final] + all_thinkings2[-1] + all_answers2[-1],
        final_instr2,
        is_sub_task=True
    )
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2_final.content}; answer - {answer2_final.content}")
    subtask_desc2['response'] = {"thinking": thinking2_final, "answer": answer2_final}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Stage 3: List P-positions with Reflexion
    reflect_inst = (
        "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. "
        "Using insights from previous attempts, try to solve the task better."
    )
    cot_reflect_instruction = (
        "Sub-task 3: Using the periodic pattern from Sub-task 2, generate the list of all n ≤ 2024 that are P-positions and cross-check a sample against the DP results. "
        + reflect_inst
    )
    cot_agent3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs3 = [taskInfo, thinking1_final, answer1_final, thinking2_final, answer2_final]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction,
        "context": ["user query", thinking1_final, answer1_final, thinking2_final, answer2_final],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent3(cot_inputs3, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max):
        feedback3, correct3 = await critic_agent3(
            [taskInfo, thinking3, answer3],
            "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'",
            i,
            is_sub_task=True
        )
        agents.append(f"Critic agent {critic_agent3.id}, feedback: {feedback3.content}; correct: {correct3.content}")
        if correct3.content == "True":
            break
        cot_inputs3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent3(cot_inputs3, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Stage 4: Count P-positions with CoT
    cot_instruction4 = (
        "Sub-task 4: Count the total number of P-positions identified in Sub-task 3, which equals the number of n ≤ 2024 for which Bob has a winning strategy."  
    )
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_instruction4,
        "context": ["user query", thinking3, answer3],
        "agent_collaboration": "CoT"
    }
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs