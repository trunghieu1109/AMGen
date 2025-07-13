async def forward_8(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    debate_instr = "Sub-task 1: Identify and characterize the losing positions in the game for the first player (Alice) based on the rules of removing 1 or 4 tokens. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    N_max = self.max_round
    all_thinking = [[] for _ in range(N_max)]
    all_answer = [[] for _ in range(N_max)]
    subtask_desc0 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instr,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking, answer = await agent([taskInfo], debate_instr, r, is_sub_task=True)
            else:
                input_infos = [taskInfo] + all_thinking[r-1] + all_answer[r-1]
                thinking, answer = await agent(input_infos, debate_instr, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            all_thinking[r].append(thinking)
            all_answer[r].append(answer)
    final_decision_instr = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    final_decision_agent = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking0, answer0 = await final_decision_agent([taskInfo] + all_thinking[-1] + all_answer[-1], "Sub-task 1: Identify losing positions." + final_decision_instr, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking0.content}; answer - {answer0.content}")
    subtask_desc0['response'] = {"thinking": thinking0, "answer": answer0}
    logs.append(subtask_desc0)
    print("Step 0: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 2: Based on the output from Sub-task 1, derive a formal recurrence relation or closed-form representation for the losing positions identified, validating the pattern of these positions."
    N_sc = self.max_sc
    cot_sc_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N_sc)]
    possible_answers = []
    possible_thinkings = []
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_sc_instruction,
        "context": ["user query", thinking0.content, answer0.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking1, answer1 = await cot_sc_agents[i]([taskInfo, thinking0, answer0], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_sc_agents[i].id}, thinking: {thinking1.content}; answer: {answer1.content}")
        possible_answers.append(answer1)
        possible_thinkings.append(thinking1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo, thinking0, answer0] + possible_thinkings + possible_answers, "Sub-task 2: Synthesize and choose the most consistent and correct solutions for the recurrence." , is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_instruction = "Sub-task 3: Compute the set of losing positions for all n ≤ 2024 using the derived recurrence or formula, and identify which initial n values correspond to losing positions for Alice (winning for Bob)."
    cot_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction,
        "context": ["user query", thinking0.content, answer0.content, thinking1.content, answer1.content],
        "agent_collaboration": "CoT"
    }
    thinking2, answer2 = await cot_agent([taskInfo, thinking0, answer0, thinking1, answer1], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 4: Sum the count of all positive integers n ≤ 2024 for which Bob has a winning strategy (i.e., n is a losing position for Alice), and verify the correctness of the final count." + reflect_inst
    cot_agent_reflect = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_reflect = self.max_round
    cot_inputs = [taskInfo, thinking2, answer2]
    subtask_desc3 = {
        "subtask_id": "subtask_1",
        "instruction": cot_reflect_instruction,
        "context": ["user query", thinking2.content, answer2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_reflect(cot_inputs, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_reflect.id}, thinking: {thinking3.content}; answer: {answer3.content}")
    critic_inst = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max_reflect):
        feedback, correct = await critic_agent([taskInfo, thinking3, answer3], "Please review and provide the limitations of provided solutions." + critic_inst, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs.extend([thinking3, answer3, feedback])
        thinking3, answer3 = await cot_agent_reflect(cot_inputs, cot_reflect_instruction, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_reflect.id}, refining, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking3, answer3, sub_tasks, agents)
    return final_answer, logs
