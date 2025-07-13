async def forward_184(self, taskInfo):
    from collections import Counter
    sub_tasks = []
    agents = []
    logs = []

    # Stage 0: SC_CoT to identify given elements
    cot_sc_instruction_0 = "Sub-task 1: Identify and classify the given elements: the Hamiltonian H = ε σ·n, the Pauli matrices σ, the unit vector n, and the energy constant ε."
    N0 = self.max_sc
    cot_agents_0 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N0)]
    possible_thinkings0 = []
    possible_answers0 = []
    subtask_desc0 = {"subtask_id":"subtask_1","instruction":cot_sc_instruction_0,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for i in range(N0):
        thinking0_i, answer0_i = await cot_agents_0[i]([taskInfo], cot_sc_instruction_0, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_0[i].id}, identification, thinking: {thinking0_i.content}; answer: {answer0_i.content}")
        possible_thinkings0.append(thinking0_i)
        possible_answers0.append(answer0_i)
    final_sc0 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr0 = "Given all the above thinking and answers, find the most consistent and correct identification of the elements."
    thinking1, answer1 = await final_sc0([taskInfo] + possible_thinkings0 + possible_answers0, "Sub-task 1: Synthesize and choose the most consistent answer." + final_instr0, is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc0['response'] = {"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc0)
    print("Step 1: ", sub_tasks[-1])

    # Stage 1: SC_CoT to derive (σ·n)^2 = I
    cot_sc_instruction_1 = "Sub-task 2: Derive the operator identity (σ·n)^2 = I by using the Pauli matrix algebra and the condition ||n|| = 1."
    N1 = self.max_sc
    cot_agents_1 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N1)]
    possible_thinkings1 = []
    possible_answers1 = []
    subtask_desc1 = {"subtask_id":"subtask_2","instruction":cot_sc_instruction_1,"context":["user query","thinking of subtask 1","answer of subtask 1"],"agent_collaboration":"SC_CoT"}
    for i in range(N1):
        thinking1_i, answer1_i = await cot_agents_1[i]([taskInfo, thinking1, answer1], cot_sc_instruction_1, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents_1[i].id}, deriving identity, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
        possible_thinkings1.append(thinking1_i)
        possible_answers1.append(answer1_i)
    final_sc1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr1 = "Given all the above thinking and answers, find the most consistent derivation of the identity."
    thinking2, answer2 = await final_sc1([taskInfo, thinking1, answer1] + possible_thinkings1 + possible_answers1, "Sub-task 2: Synthesize and choose the most consistent derivation." + final_instr1, is_sub_task=True)
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc1['response'] = {"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc1)
    print("Step 2: ", sub_tasks[-1])

    # Stage 2: Reflexion to solve eigenvalues ±1
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 3: Solve the eigenvalue equation for σ·n using the identity (σ·n)^2 = I to find its eigenvalues ±1." + reflect_inst
    cot_agent3 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent3 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs3 = [taskInfo, thinking2, answer2]
    subtask_desc2 = {"subtask_id":"subtask_3","instruction":cot_reflect_instruction,"context":["user query","thinking of subtask 2","answer of subtask 2"],"agent_collaboration":"Reflexion"}
    thinking3, answer3 = await cot_agent3(cot_inputs3, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, initial solution, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(self.max_round):
        critic_inst = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
        feedback3, correct3 = await critic_agent3([taskInfo, thinking3, answer3], critic_inst, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent3.id}, feedback, thinking: {feedback3.content}; answer: {correct3.content}")
        if correct3.content == "True":
            break
        cot_inputs3.extend([thinking3, answer3, feedback3])
        thinking3, answer3 = await cot_agent3(cot_inputs3, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, refined solution, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc2['response'] = {"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc2)
    print("Step 3: ", sub_tasks[-1])

    # Stage 3: Debate to scale by ε and choose +ε, -ε
    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction = "Sub-task 4: Scale the eigenvalues of σ·n by ε to obtain H’s eigenvalues ±ε and select the matching choice from the provided options." + debate_instr
    debate_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking4 = [[] for _ in range(self.max_round)]
    all_answer4 = [[] for _ in range(self.max_round)]
    subtask_desc3 = {"subtask_id":"subtask_4","instruction":debate_instruction,"context":["user query","thinking of subtask 3","answer of subtask 3"],"agent_collaboration":"Debate"}
    for r in range(self.max_round):
        for agent in debate_agents:
            if r == 0:
                thinking4_i, answer4_i = await agent([taskInfo, thinking3, answer3], debate_instruction, r, is_sub_task=True)
            else:
                inputs4 = [taskInfo, thinking3, answer3] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4_i, answer4_i = await agent(inputs4, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking4_i.content}; answer: {answer4_i.content}")
            all_thinking4[r].append(thinking4_i)
            all_answer4[r].append(answer4_i)
    final_decision4 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr4 = "Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking4, answer4 = await final_decision4([taskInfo, thinking3, answer3] + all_thinking4[-1] + all_answer4[-1], "Sub-task 4: " + final_instr4, is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc3['response'] = {"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc3)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs