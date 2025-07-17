async def forward_166(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Sub-task 1: Compute normalization constant N via SC_CoT
    cot_sc_instruction = "Sub-task 1: Compute N = sqrt(1 + sin(2*phi)*exp(-2*alpha**2)) for phi=-pi/4, alpha=0.5 with numeric evaluation."
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_thinkings = []
    possible_answers = []
    sub_desc = {"subtask_id":"subtask_1","instruction":cot_sc_instruction,"agent_collaboration":"SC_CoT"}
    for agent in cot_agents:
        thinking, answer = await agent([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings.append(thinking)
        possible_answers.append(answer)
    final_decision = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision([taskInfo] + possible_thinkings + possible_answers, "Sub-task 1: Synthesize N result.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: {answer1.content}")
    sub_desc['response'] = {"thinking":thinking1,"answer":answer1}
    logs.append(sub_desc)
    print("Step 1: ", sub_tasks[-1])
    N_value = float(answer1.content)

    # Sub-task 2: Construct rho in truncated Fock basis via SC_CoT
    cot_sc_instruction = "Sub-task 2: Build rho = |psi><psi| in Fock basis up to n=10 using N_value, phi, alpha."  
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_thinkings = []
    possible_answers = []
    sub_desc = {"subtask_id":"subtask_2","instruction":cot_sc_instruction,"agent_collaboration":"SC_CoT"}
    for agent in cot_agents:
        thinking, answer = await agent([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking.content}; answer: {answer.content}")
        possible_thinkings.append(thinking)
        possible_answers.append(answer)
    final_decision = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking2, answer2 = await final_decision([taskInfo, thinking1, answer1] + possible_thinkings + possible_answers, "Sub-task 2: Synthesize rho matrix.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: {answer2.content}")
    sub_desc['response'] = {"thinking":thinking2,"answer":answer2}
    logs.append(sub_desc)
    print("Step 2: ", sub_tasks[-1])
    rho_matrix = answer2.content

    # Sub-task 3: Compute moments of rho via Debate
    debate_instruction = "Sub-task 3: Numerically compute <x>,<p>,Var(x),Var(p),Cov(x,p) of rho in truncated basis." + "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinkings = []
    all_answers = []
    sub_desc = {"subtask_id":"subtask_3","instruction":debate_instruction,"agent_collaboration":"Debate"}
    for r in range(self.max_round):
        round_thinkings = []
        round_answers = []
        for agent in debate_agents:
            if r==0:
                thinking, answer = await agent([taskInfo, thinking2, answer2], debate_instruction, r, is_sub_task=True)
            else:
                thinking, answer = await agent([taskInfo, thinking2, answer2] + all_thinkings[-1] + all_answers[-1], debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            round_thinkings.append(thinking)
            round_answers.append(answer)
        all_thinkings.append(round_thinkings)
        all_answers.append(round_answers)
    final_decision = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision([taskInfo, thinking2, answer2] + all_thinkings[-1] + all_answers[-1], "Sub-task 3: Provide final moments.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: {answer3.content}")
    sub_desc['response'] = {"thinking":thinking3,"answer":answer3}
    logs.append(sub_desc)
    print("Step 3: ", sub_tasks[-1])
    moments = answer3.content

    # Sub-task 4: Build reference Gaussian tau via Reflexion
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 4: Construct reference Gaussian state tau from moments." + reflect_inst
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    inputs = [taskInfo, thinking3, answer3]
    sub_desc = {"subtask_id":"subtask_4","instruction":cot_reflect_instruction,"agent_collaboration":"Reflexion"}
    thinking4, answer4 = await cot_agent(inputs, cot_reflect_instruction, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    for _ in range(self.max_round):
        feedback, correct = await critic_agent([taskInfo, thinking4, answer4], "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'.", is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content.strip() == "True": break
        inputs += [thinking4, answer4, feedback]
        thinking4, answer4 = await cot_agent(inputs, cot_reflect_instruction, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: {answer4.content}")
    sub_desc['response'] = {"thinking":thinking4,"answer":answer4}
    logs.append(sub_desc)
    print("Step 4: ", sub_tasks[-1])
    tau_matrix = answer4.content

    # Sub-task 5: Diagonalize rho and tau via Debate
    debate_instruction = "Sub-task 5: Numerically diagonalize rho and tau to get eigenvalues." + "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinkings = []
    all_answers = []
    sub_desc = {"subtask_id":"subtask_5","instruction":debate_instruction,"agent_collaboration":"Debate"}
    for r in range(self.max_round):
        round_thinkings = []
        round_answers = []
        for agent in debate_agents:
            if r==0:
                thinking, answer = await agent([taskInfo, thinking2, answer2, thinking4, answer4], debate_instruction, r, is_sub_task=True)
            else:
                thinking, answer = await agent([taskInfo, thinking2, answer2, thinking4, answer4] + all_thinkings[-1] + all_answers[-1], debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking.content}; answer: {answer.content}")
            round_thinkings.append(thinking)
            round_answers.append(answer)
        all_thinkings.append(round_thinkings)
        all_answers.append(round_answers)
    final_decision = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking5, answer5 = await final_decision([taskInfo, thinking2, answer2, thinking4, answer4] + all_thinkings[-1] + all_answers[-1], "Sub-task 5: Provide eigenvalues.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 5 output: {answer5.content}")
    sub_desc['response'] = {"thinking":thinking5,"answer":answer5}
    logs.append(sub_desc)
    print("Step 5: ", sub_tasks[-1])
    eigvals_rho, eigvals_tau = answer5.content

    # Sub-task 6: Compute delta_b via CoT
    cot_instruction = "Sub-task 6: Compute delta_b = Tr(rho ln rho) - Tr(tau ln tau) using eigenvalues, then compare to choices and select closest."  
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    thinking6, answer6 = await cot_agent([taskInfo, thinking5, answer5], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: {answer6.content}")
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs