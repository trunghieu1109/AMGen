async def forward_186(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_instruction = "Sub-task 1: Extract and summarize essential inputs: the list of target stars (Canopus, Polaris with known V, and four M_V=15 stars at given distances), the detectability criterion S/N≥10 per binned pixel in a 1h exposure, and the need for detailed ESPRESSO parameters."
    cot_agent = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":cot_instruction,"context":["user query"],"agent_collaboration":"CoT"}
    thinking1, answer1 = await cot_agent([taskInfo], cot_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent.id}, extracting inputs, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking":thinking1, "answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction = "Sub-task 2: Compute apparent V magnitudes for the four hypothetical M_V=15 stars at distances 5, 10, 50, and 200 pc using m_V=M_V+5 log10(d/10pc) and verify these calculations."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":cot_sc_instruction,"context":["user query","Sub-task 1 output"],"agent_collaboration":"SC_CoT"}
    possible_thinkings = []
    possible_answers = []
    for i in range(N):
        thinking2_i, answer2_i = await cot_agents[i]([taskInfo, thinking1, answer1], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, computing m_V values, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_thinkings.append(thinking2_i)
        possible_answers.append(answer2_i)
    final_decision_agent_2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    synth_instr2 = "Sub-task 2: Synthesize and choose the most consistent and correct apparent magnitude calculations."
    thinking2, answer2 = await final_decision_agent_2([taskInfo, thinking1, answer1] + possible_thinkings + possible_answers, synth_instr2, is_sub_task=True)
    agents.append(f"Final Decision Agent {final_decision_agent_2.id}, synthesizing m_V results, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking":thinking2, "answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    debate_instr = "Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_instruction_3 = "Sub-task 3: Retrieve and summarize official ESPRESSO+VLT performance parameters from ESO documentation or ETC: spectral resolution, throughput vs. wavelength, pixel sampling, telescope collecting area, detector efficiency. " + debate_instr
    debate_agents3 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking3 = [[] for _ in range(self.max_round)]
    all_answer3 = [[] for _ in range(self.max_round)]
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":debate_instruction_3,"context":["user query","Sub-task 2 output"],"agent_collaboration":"Debate"}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents3):
            if r == 0:
                thinking3, answer3 = await agent([taskInfo, thinking2, answer2], debate_instruction_3, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3, answer3 = await agent(inputs, debate_instruction_3, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3.content}; answer: {answer3.content}")
            all_thinking3[r].append(thinking3)
            all_answer3[r].append(answer3)
    final_decision_agent_3 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final3_instr = "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking3_f, answer3_f = await final_decision_agent_3([taskInfo] + all_thinking3[-1] + all_answer3[-1], final3_instr, is_sub_task=True)
    agents.append(f"Final Decision Agent {final_decision_agent_3.id}, summarizing instrument parameters, thinking: {thinking3_f.content}; answer: {answer3_f.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3_f.content}; answer - {answer3_f.content}")
    subtask_desc3['response'] = {"thinking":thinking3_f, "answer":answer3_f}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    debate_instruction_4 = "Sub-task 4: Perform first-principles photon-counting S/N estimation per binned pixel in a 1h exposure for each star: convert m_V to photon flux, apply telescope area, throughput, resolution, pixel width, exposure time, and compute S/N ∝ sqrt(N_photons)." + debate_instr
    debate_agents4 = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking4 = [[] for _ in range(self.max_round)]
    all_answer4 = [[] for _ in range(self.max_round)]
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":debate_instruction_4,"context":["user query","Sub-task 3 output"],"agent_collaboration":"Debate"}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents4):
            if r == 0:
                thinking4, answer4 = await agent([taskInfo, thinking3_f, answer3_f], debate_instruction_4, r, is_sub_task=True)
            else:
                inputs = [taskInfo, thinking3_f, answer3_f] + all_thinking4[r-1] + all_answer4[r-1]
                thinking4, answer4 = await agent(inputs, debate_instruction_4, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent_4 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final4_instr = "Sub-task 4: Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking4_f, answer4_f = await final_decision_agent_4([taskInfo] + all_thinking4[-1] + all_answer4[-1], final4_instr, is_sub_task=True)
    agents.append(f"Final Decision Agent {final_decision_agent_4.id}, summarizing S/N estimates, thinking: {thinking4_f.content}; answer: {answer4_f.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4_f.content}; answer - {answer4_f.content}")
    subtask_desc4['response'] = {"thinking":thinking4_f, "answer":answer4_f}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 5: Fact-check the S/N estimates against official ESPRESSO ETC results or published performance tables; if discrepancies exceed tolerance, iteratively adjust calculation parameters." + reflect_inst
    cot_agent5 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent5 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    N_max = self.max_round
    cot_inputs5 = [taskInfo, thinking4_f, answer4_f]
    subtask_desc5 = {"subtask_id":"subtask_5","instruction":cot_reflect_instruction,"context":["user query","Sub-task 4 thinking","Sub-task 4 answer"],"agent_collaboration":"Reflexion"}
    thinking5, answer5 = await cot_agent5(cot_inputs5, cot_reflect_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent5.id}, initial fact-check, thinking: {thinking5.content}; answer: {answer5.content}")
    for i in range(N_max):
        critic_inst = "Please review the answer above and criticize on where might be wrong. If you are absolutely sure it is correct, output exactly 'True' in 'correct'"
        feedback5, correct5 = await critic_agent5([taskInfo, thinking5, answer5], critic_inst, i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent5.id}, feedback: {feedback5.content}; correct: {correct5.content}")
        if correct5.content == "True":
            break
        cot_inputs5.extend([thinking5, answer5, feedback5])
        thinking5, answer5 = await cot_agent5(cot_inputs5, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent5.id}, refinement {i+1}, thinking: {thinking5.content}; answer: {answer5.content}")
    sub_tasks.append(f"Sub-task 5 output: thinking - {thinking5.content}; answer - {answer5.content}")
    subtask_desc5['response'] = {"thinking":thinking5, "answer":answer5}
    logs.append(subtask_desc5)
    print("Step 5: ", sub_tasks[-1])

    cot6_instruction = "Sub-task 6: Classify each star as detectable (S/N≥10) or not based on the validated S/N values and count the total detectable stars."
    cot_agent6 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc6 = {"subtask_id":"subtask_6","instruction":cot6_instruction,"context":["user query","Sub-task 5 output"],"agent_collaboration":"CoT"}
    thinking6, answer6 = await cot_agent6([taskInfo, thinking5, answer5], cot6_instruction, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent6.id}, classification, thinking: {thinking6.content}; answer: {answer6.content}")
    sub_tasks.append(f"Sub-task 6 output: thinking - {thinking6.content}; answer - {answer6.content}")
    subtask_desc6['response'] = {"thinking":thinking6, "answer":answer6}
    logs.append(subtask_desc6)
    print("Step 6: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking6, answer6, sub_tasks, agents)
    return final_answer, logs