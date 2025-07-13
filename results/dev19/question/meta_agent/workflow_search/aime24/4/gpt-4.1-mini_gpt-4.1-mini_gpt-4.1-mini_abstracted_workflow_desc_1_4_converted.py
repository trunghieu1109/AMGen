async def forward_4(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Stage 1: Characterize primes and find minimal prime p

    # Sub-task 1: Analyze condition n^4 + 1 ≡ 0 (mod p) to characterize primes p
    debate_instr_1 = "Sub-task 1: Analyze the condition n^4 + 1 ≡ 0 (mod p) to characterize primes p for which there exists a positive integer n satisfying n^4 ≡ -1 (mod p). Carefully determine necessary and sufficient conditions on p for -1 to be a quartic residue modulo p. Avoid assuming any properties without proof or explicit reasoning. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_1 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_1 = self.max_round
    all_thinking_1 = [[] for _ in range(N_max_1)]
    all_answer_1 = [[] for _ in range(N_max_1)]
    subtask_desc1 = {
        "subtask_id": "subtask_1",
        "instruction": debate_instr_1,
        "context": ["user query"],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_1):
        for i, agent in enumerate(debate_agents_1):
            if r == 0:
                thinking1, answer1 = await agent([taskInfo], debate_instr_1, r, is_sub_task=True)
            else:
                input_infos_1 = [taskInfo] + all_thinking_1[r-1]
                thinking1, answer1 = await agent(input_infos_1, debate_instr_1, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing quartic residue condition, thinking: {thinking1.content}; answer: {answer1.content}")
            all_thinking_1[r].append(thinking1)
            all_answer_1[r].append(answer1)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_1 = "Sub-task 1: Given all the above thinking and answers, reason over them carefully and provide a final characterization of primes p for which -1 is a quartic residue modulo p."
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + all_thinking_1[-1], final_instr_1, is_sub_task=True)
    agents.append(f"Final Decision agent, Sub-task 1, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Determine minimal prime p satisfying n^4 ≡ -1 (mod p)
    cot_sc_instruction_2 = "Sub-task 2: Based on the characterization from Sub-task 1, determine the minimal prime p for which there exists a positive integer n with n^4 + 1 divisible by p (i.e., n^4 ≡ -1 mod p). Explicitly check candidate primes and justify minimality."
    N_sc = self.max_sc
    cot_agents_2 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc)]
    possible_answers_2 = []
    possible_thinkings_2 = []
    subtask_desc2 = {
        "subtask_id": "subtask_2",
        "instruction": cot_sc_instruction_2,
        "context": ["user query", thinking1.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc):
        thinking2, answer2 = await cot_agents_2[i]([taskInfo, thinking1], cot_sc_instruction_2, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_2[i].id}, determining minimal prime p, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers_2.append(answer2)
        possible_thinkings_2.append(thinking2)
    final_decision_agent_2 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_2 = "Sub-task 2: Given all the above thinking and answers, synthesize and choose the minimal prime p satisfying the condition."
    thinking2, answer2 = await final_decision_agent_2([taskInfo] + possible_thinkings_2, final_instr_2, is_sub_task=True)
    agents.append(f"Final Decision agent, Sub-task 2, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: Analyze lifting conditions modulo p^2
    debate_instr_3 = "Sub-task 3: Analyze the conditions for lifting solutions from modulo p to modulo p^2. Derive necessary criteria under which a solution n modulo p to n^4 + 1 ≡ 0 (mod p) can be lifted to modulo p^2. Include the role of the derivative f'(n) and the linear congruence for the correction term k in the lift n + p k. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_3 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_3 = self.max_round
    all_thinking_3 = [[] for _ in range(N_max_3)]
    all_answer_3 = [[] for _ in range(N_max_3)]
    subtask_desc3 = {
        "subtask_id": "subtask_3",
        "instruction": debate_instr_3,
        "context": ["user query", thinking2.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_3):
        for i, agent in enumerate(debate_agents_3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2], debate_instr_3, r, is_sub_task=True)
            else:
                input_infos_3 = [taskInfo, thinking2] + all_thinking_3[r-1]
                thinking3, answer3 = await agent(input_infos_3, debate_instr_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, analyzing lifting conditions, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking_3[r].append(thinking3)
            all_answer_3[r].append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_3 = "Sub-task 3: Given all the above thinking and answers, synthesize the necessary and sufficient conditions for lifting solutions modulo p to modulo p^2."
    thinking3, answer3 = await final_decision_agent_3([taskInfo, thinking2] + all_thinking_3[-1], final_instr_3, is_sub_task=True)
    agents.append(f"Final Decision agent, Sub-task 3, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: Confirm least prime p for which p^2 divides n^4 + 1
    cot_sc_instruction_4 = "Sub-task 4: Using the lifting conditions from Sub-task 3, confirm the least prime p for which there exists a positive integer n such that p^2 divides n^4 + 1. Provide explicit reasoning and verification that no smaller prime satisfies the stronger divisibility condition."
    N_sc_4 = self.max_sc
    cot_agents_4 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc_4)]
    possible_answers_4 = []
    possible_thinkings_4 = []
    subtask_desc4 = {
        "subtask_id": "subtask_4",
        "instruction": cot_sc_instruction_4,
        "context": ["user query", thinking3.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_4):
        thinking4, answer4 = await cot_agents_4[i]([taskInfo, thinking3], cot_sc_instruction_4, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_4[i].id}, confirming least prime p for p^2 divisibility, thinking: {thinking4.content}; answer: {answer4.content}")
        possible_answers_4.append(answer4)
        possible_thinkings_4.append(thinking4)
    final_decision_agent_4 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_4 = "Sub-task 4: Given all the above thinking and answers, synthesize and confirm the least prime p for which p^2 divides n^4 + 1."
    thinking4, answer4 = await final_decision_agent_4([taskInfo] + possible_thinkings_4, final_instr_4, is_sub_task=True)
    agents.append(f"Final Decision agent, Sub-task 4, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    # Extract prime p from Sub-task 4 answer (assumed to be explicit in answer4.content)
    # For safety, parse prime p from answer4.content
    import re
    p_match = re.search(r'\b(\d+)\b', answer4.content)
    if p_match:
        p = int(p_match.group(1))
    else:
        p = None

    # Stage 2: Find solutions modulo p and lift to modulo p^2

    # Sub-task 5: Find all positive integers n modulo p such that n^4 + 1 ≡ 0 (mod p)
    cot_instruction_5 = f"Sub-task 5: For the identified prime p={p}, find all positive integers n modulo p such that n^4 + 1 ≡ 0 (mod p). Explicitly list these solutions and verify correctness."
    cot_agent_5 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc5 = {
        "subtask_id": "subtask_5",
        "instruction": cot_instruction_5,
        "context": ["user query", thinking4.content],
        "agent_collaboration": "CoT"
    }
    thinking5, answer5 = await cot_agent_5([taskInfo, thinking4], cot_instruction_5, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent_5.id}, finding solutions modulo p, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking": thinking5, "answer": answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    # Parse solutions n0 modulo p from answer5.content
    # Expect answer5.content to contain explicit list of solutions modulo p
    # Extract all integers between 1 and p-1 that satisfy condition
    import ast
    solutions_mod_p = []
    try:
        # Attempt to parse list from answer5.content
        # Look for pattern like [a, b, c]
        list_match = re.search(r'\[(.*?)\]', answer5.content)
        if list_match:
            list_str = '[' + list_match.group(1) + ']'
            solutions_mod_p = ast.literal_eval(list_str)
            # Filter to positive integers modulo p
            solutions_mod_p = [x for x in solutions_mod_p if isinstance(x, int) and 1 <= x < p]
    except Exception:
        solutions_mod_p = []

    # Sub-task 6: Lift each solution n modulo p to modulo p^2 by solving linear congruence for k
    debate_instr_6 = f"Sub-task 6: For each solution n modulo p found in Sub-task 5, lift it to a solution modulo p^2 by explicitly setting up and solving the linear congruence f(n + p k) ≡ 0 (mod p^2) for the correction term k. Perform detailed step-by-step arithmetic expansions and solve for k without assuming k=0. Document all intermediate calculations and ensure uniqueness of the lifted solution when f'(n) ≠ 0. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents_6 = [LLMAgentBase(["thinking", "answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.0) for role in self.debate_role]
    N_max_6 = self.max_round
    all_thinking_6 = [[] for _ in range(N_max_6)]
    all_answer_6 = [[] for _ in range(N_max_6)]
    subtask_desc6 = {
        "subtask_id": "subtask_6",
        "instruction": debate_instr_6,
        "context": ["user query", thinking5.content],
        "agent_collaboration": "Debate"
    }
    for r in range(N_max_6):
        for i, agent in enumerate(debate_agents_6):
            if r == 0:
                thinking6, answer6 = await agent([taskInfo, thinking5], debate_instr_6, r, is_sub_task=True)
            else:
                input_infos_6 = [taskInfo, thinking5] + all_thinking_6[r-1]
                thinking6, answer6 = await agent(input_infos_6, debate_instr_6, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, lifting solutions modulo p^2, thinking: {thinking6.content}; answer: {answer6.content}")
            all_thinking_6[r].append(thinking6)
            all_answer_6[r].append(answer6)
    final_decision_agent_6 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_6 = "Sub-task 6: Given all the above thinking and answers, synthesize the explicit lifted solutions modulo p^2 with detailed arithmetic and correction term k."
    thinking6, answer6 = await final_decision_agent_6([taskInfo, thinking5] + all_thinking_6[-1], final_instr_6, is_sub_task=True)
    agents.append(f"Final Decision agent, Sub-task 6, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking": thinking6, "answer": answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    # Sub-task 7: Verify each lifted solution m modulo p^2 by substitution
    reflect_inst_7 = "Sub-task 7: Verify each lifted solution m modulo p^2 obtained in Sub-task 6 by substituting back into m^4 + 1 modulo p^2 to confirm divisibility. Explicitly perform modular arithmetic verification for each candidate m and discard invalid solutions. This ensures correctness and prevents errors from assumptions or arithmetic omissions. Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction_7 = "Sub-task 7: Your problem is to verify lifted solutions modulo p^2." + reflect_inst_7
    cot_agent_7 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent_7 = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max_7 = self.max_round
    cot_inputs_7 = [taskInfo, thinking6]
    subtask_desc7 = {
        "subtask_id": "subtask_7",
        "instruction": cot_reflect_instruction_7,
        "context": ["user query", thinking6.content],
        "agent_collaboration": "Reflexion"
    }
    thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent_7.id}, verifying lifted solutions, thinking: {thinking7.content}; answer: {answer7.content}")
    critic_inst_7 = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
    for i in range(N_max_7):
        feedback7, correct7 = await critic_agent_7([taskInfo, thinking7], critic_inst_7, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent_7.id}, providing feedback, thinking: {feedback7.content}; answer: {correct7.content}")
        if correct7.content.strip() == "True":
            break
        cot_inputs_7.extend([thinking7, feedback7])
        thinking7, answer7 = await cot_agent_7(cot_inputs_7, cot_reflect_instruction_7, i + 1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent_7.id}, refining verification, thinking: {thinking7.content}; answer: {answer7.content}")
    sub_tasks.append(f"Sub-task 7 output: thinking - {thinking7.content}; answer - {answer7.content}")
    subtask_desc7['response'] = {"thinking": thinking7, "answer": answer7}
    logs.append(subtask_desc7)
    print("Step 7: ", sub_tasks[-1])

    # Sub-task 8: Determine least positive integer m such that m^4 + 1 divisible by p^2
    cot_sc_instruction_8 = "Sub-task 8: Among the verified lifted solutions modulo p^2, determine the least positive integer m such that m^4 + 1 is divisible by p^2. Provide clear justification for minimality and confirm positivity. Present the final answer with all supporting arithmetic and verification details."
    N_sc_8 = self.max_sc
    cot_agents_8 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0) for _ in range(N_sc_8)]
    possible_answers_8 = []
    possible_thinkings_8 = []
    subtask_desc8 = {
        "subtask_id": "subtask_8",
        "instruction": cot_sc_instruction_8,
        "context": ["user query", thinking7.content],
        "agent_collaboration": "SC_CoT"
    }
    for i in range(N_sc_8):
        thinking8, answer8 = await cot_agents_8[i]([taskInfo, thinking7], cot_sc_instruction_8, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_8[i].id}, determining least positive m, thinking: {thinking8.content}; answer: {answer8.content}")
        possible_answers_8.append(answer8)
        possible_thinkings_8.append(thinking8)
    final_decision_agent_8 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr_8 = "Sub-task 8: Given all the above thinking and answers, synthesize and provide the least positive integer m such that m^4 + 1 is divisible by p^2 with full justification."
    thinking8, answer8 = await final_decision_agent_8([taskInfo] + possible_thinkings_8, final_instr_8, is_sub_task=True)
    agents.append(f"Final Decision agent, Sub-task 8, thinking: {thinking8.content}; answer: {answer8.content}")
    sub_tasks.append(f"Sub-task 8 output: thinking - {thinking8.content}; answer - {answer8.content}")
    subtask_desc8['response'] = {"thinking": thinking8, "answer": answer8}
    logs.append(subtask_desc8)
    print("Step 8: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking8, answer8, sub_tasks, agents)
    return final_answer, logs
