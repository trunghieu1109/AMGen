async def forward_27(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction_1 = (
        "Sub-task 1: Identify and clearly state the domain of the problem: define N as a four-digit integer with digits d1, d2, d3, d4, "
        "where 1000 ≤ N ≤ 9999 and d1 ≠ 0. Explicitly represent N in terms of its digits and clarify the range of each digit.")
    cot_agent_1 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": cot_instruction_1,
        "context": ["user query"],
        "agent_collaboration": "CoT"
    }
    thinking1, answer1 = await cot_agent_1([taskInfo], cot_instruction_1, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_1.id}, analyzing domain and digits, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    debate_instruction_2 = (
        "Sub-task 2: Formulate the condition that changing any single digit of N to 1 results in a number divisible by 7. "
        "Express these four divisibility conditions explicitly as modular arithmetic constraints, one for each digit position, ensuring clarity and completeness. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.")
    debate_agents_2 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_2 = self.max_round
    all_thinking2 = [[] for _ in range(N_max_2)]
    all_answer2 = [[] for _ in range(N_max_2)]
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": debate_instruction_2,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_2):
        for i, agent in enumerate(debate_agents_2):
            if r == 0:
                thinking2, answer2 = await agent([taskInfo, thinking1], debate_instruction_2, r, is_sub_task=True)
            else:
                input_infos_2 = [taskInfo, thinking1] + all_thinking2[r-1]
                thinking2, answer2 = await agent(input_infos_2, debate_instruction_2, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, formulating modular conditions, thinking: {thinking2.content}; answer: {answer2.content}")
            all_thinking2[r].append(thinking2)
            all_answer2[r].append(answer2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_2 = "Sub-task 2: Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + all_thinking2[-1], debate_instruction_2 + final_instr_2, is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing modular conditions, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    reflect_inst_3 = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_3 = (
        "Sub-task 3: Clarify all assumptions and constraints related to digit substitution, including: handling cases where the digit is already 1 (the substitution is trivial), "
        "ensuring no leading zeros after substitution, and confirming that the decimal representation remains valid. Explicitly state these assumptions to avoid ambiguity in later steps. "
        + reflect_inst_3)
    cot_agent_3 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_3 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_3 = self.max_round
    cot_inputs_3 = [taskInfo, thinking1, thinking2]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": cot_reflect_instruction_3,
        "context": ["user query", thinking1.content, thinking2.content],
        "agent_collaboration": "Reflexion"
    }
    thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_3.id}, clarifying assumptions, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max_3):
        feedback, correct = await critic_agent_3([taskInfo, thinking3], "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_3.id}, providing feedback, thinking: {feedback.content}; answer: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs_3.extend([thinking3, feedback])
        thinking3, answer3 = await cot_agent_3(cot_inputs_3, cot_reflect_instruction_3, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_3.id}, refining assumptions, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    debate_instruction_4 = (
        "Sub-task 4: Analyze the modular arithmetic constraints derived in stage_0 to reduce the search space for digits d1, d2, d3, d4. "
        "Identify necessary congruences and relationships between digits that must hold for the divisibility conditions, and summarize these constraints to guide enumeration. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.")
    debate_agents_4 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_4 = self.max_round
    all_thinking4 = [[] for _ in range(N_max_4)]
    all_answer4 = [[] for _ in range(N_max_4)]
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": debate_instruction_4,
        "context": ["user query", thinking2.content, thinking3.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_4):
        for i, agent in enumerate(debate_agents_4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking2, thinking3], debate_instruction_4, r, is_sub_task=True)
            else:
                input_infos_4 = [taskInfo, thinking2, thinking3] + all_thinking4[r-1]
                thinking4, answer4 = await agent(input_infos_4, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing modular constraints, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_4 = "Sub-task 4: Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking4, answer4 = await final_decision_agent_4([taskInfo, thinking2, thinking3] + all_thinking4[-1], debate_instruction_4 + final_instr_4, is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing modular constraints analysis, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    debate_instruction_5 = (
        "Sub-task 5: Systematically enumerate all four-digit numbers N = d1 d2 d3 d4 that satisfy the modular constraints identified in subtask_4. "
        "For each candidate, generate all four digit-substituted numbers (changing each digit to 1) and explicitly check divisibility by 7 for each. "
        "Collect all candidates that satisfy all four divisibility conditions without exception. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer. "
        "Embed code snippets or pseudocode to guide systematic enumeration and verification.")
    debate_agents_5 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_5 = self.max_round
    all_thinking5 = [[] for _ in range(N_max_5)]
    all_answer5 = [[] for _ in range(N_max_5)]
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": debate_instruction_5,
        "context": ["user query", thinking4.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_5):
        for i, agent in enumerate(debate_agents_5):
            if r == 0:
                thinking5, answer5 = await agent([taskInfo, thinking4], debate_instruction_5, r, is_sub_task=True)
            else:
                input_infos_5 = [taskInfo, thinking4] + all_thinking5[r-1]
                thinking5, answer5 = await agent(input_infos_5, debate_instruction_5, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, enumerating and verifying candidates, thinking: {thinking5.content}; answer: {answer5.content}")
            all_thinking5[r].append(thinking5)
            all_answer5[r].append(answer5)
    final_decision_agent_5 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_5 = "Sub-task 5: Given all the above thinking and answers, reason over them carefully and provide the greatest valid number N satisfying all conditions."
    thinking5, answer5 = await final_decision_agent_5([taskInfo, thinking4] + all_thinking5[-1], debate_instruction_5 + final_instr_5, is_sub_task=True)
    agents.append(f"Final Decision agent, selecting greatest valid N, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    debate_instruction_6 = (
        "Sub-task 6: Perform a rigorous validation of the candidate numbers obtained from subtask_5 by cross-checking each candidate against the original problem conditions. "
        "Confirm that each candidate’s four digit-substituted numbers are divisible by 7, and resolve any conflicts or discrepancies. "
        "Identify the greatest valid candidate N from this validated set. "
        "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer.")
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking6 = [[] for _ in range(N_max_6)]
    all_answer6 = [[] for _ in range(N_max_6)]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": debate_instruction_6,
        "context": ["user query", thinking5.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking5], debate_instruction_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking5] + all_thinking6[r-1]
                thinking6, answer6 = await agent(input_infos_6, debate_instruction_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, validating candidates, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking6[r].append(thinking6)
            all_answer6[r].append(answer6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_6 = "Sub-task 6: Given all the above thinking and answers, reason over them carefully and provide the greatest valid candidate N after validation."
    thinking6, answer6 = await final_decision_agent_6([taskInfo, thinking5] + all_thinking6[-1], debate_instruction_6 + final_instr_6, is_sub_task=True)
    agents.append(f"Final Decision agent, finalizing greatest valid N after validation, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    cot_instruction_7 = (
        "Sub-task 7: Express the validated greatest number N in terms of quotient Q and remainder R when divided by 1000, i.e., compute Q = floor(N/1000) and R = N mod 1000. "
        "Clearly identify Q and R from N.")
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_instruction_7,
        "context": ["user query", thinking6.content],
        "agent_collaboration": "CoT"
    }
    thinking7, answer7 = await cot_agent_7([taskInfo, thinking6], cot_instruction_7, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_7.id}, computing Q and R, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    cot_instruction_8 = (
        "Sub-task 8: Compute the sum Q + R as required by the problem and present the final answer. "
        "Before outputting, perform a final verification step to ensure that Q and R correspond correctly to the validated greatest N and that the sum is accurate.")
    cot_agent_8 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_instruction_8,
        "context": ["user query", thinking7.content],
        "agent_collaboration": "CoT"
    }
    thinking8, answer8 = await cot_agent_8([taskInfo, thinking7], cot_instruction_8, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_8.id}, computing final sum Q+R, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs
