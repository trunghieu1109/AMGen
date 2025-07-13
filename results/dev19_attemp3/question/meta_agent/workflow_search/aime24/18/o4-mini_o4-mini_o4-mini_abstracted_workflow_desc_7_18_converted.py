async def forward_18(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    cot_sc_instruction = "Sub-task 1: Parameterize line AB and the family F of unit segments PQ with P on the x-axis and Q on the y-axis."
    cot_agents = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc1 = {"subtask_id":"subtask_1","instruction":cot_sc_instruction,"context":["user query"],"agent_collaboration":"SC_CoT"}
    for agent in cot_agents:
        thinking1_i, answer1_i = await agent([taskInfo], cot_sc_instruction, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
        possible_thinkings.append(thinking1_i)
        possible_answers.append(answer1_i)
    synth_agent1 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    synth_instr1 = "Sub-task 1: Synthesize and choose the consistent parameterizations."
    thinking1, answer1 = await synth_agent1([taskInfo] + possible_thinkings + possible_answers, synth_instr1, is_sub_task=True)
    agents.append(f"Final Decision Agent, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    cot_sc_instruction2 = "Sub-task 2: Derive the equation and constraints under which a general point on AB lies on a segment from F distinct from AB."
    cot_agents2 = [LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(self.max_sc)]
    possible_thinkings2 = []
    possible_answers2 = []
    subtask_desc2 = {"subtask_id":"subtask_2","instruction":cot_sc_instruction2,"context":["user query",thinking1,answer1],"agent_collaboration":"SC_CoT"}
    for agent in cot_agents2:
        thinking2_i, answer2_i = await agent([taskInfo, thinking1, answer1], cot_sc_instruction2, is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, thinking: {thinking2_i.content}; answer: {answer2_i.content}")
        possible_thinkings2.append(thinking2_i)
        possible_answers2.append(answer2_i)
    synth_agent2 = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    synth_instr2 = "Sub-task 2: Synthesize and choose the consistent membership condition."
    thinking2, answer2 = await synth_agent2([taskInfo, thinking1, answer1] + possible_thinkings2 + possible_answers2, synth_instr2, is_sub_task=True)
    agents.append(f"Final Decision Agent, thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    debate_instruction = "Sub-task 3: Identify the unique point C on AB not on any other segment in F. Given solutions to the problem from other agents, consider their opinions as additional advice. Please think carefully and provide an updated answer."
    debate_agents = [LLMAgentBase(["thinking","answer"], "Debate Agent", model=self.node_model, role=role, temperature=0.5) for role in self.debate_role]
    all_thinking3 = [[] for _ in range(self.max_round)]
    all_answer3 = [[] for _ in range(self.max_round)]
    subtask_desc3 = {"subtask_id":"subtask_3","instruction":debate_instruction,"context":["user query",thinking1,answer1,thinking2,answer2],"agent_collaboration":"Debate"}
    for r in range(self.max_round):
        for i, agent in enumerate(debate_agents):
            if r == 0:
                thinking3_i, answer3_i = await agent([taskInfo, thinking1, answer1, thinking2, answer2], debate_instruction, r, is_sub_task=True)
            else:
                input_prev = [taskInfo, thinking1, answer1, thinking2, answer2] + all_thinking3[r-1] + all_answer3[r-1]
                thinking3_i, answer3_i = await agent(input_prev, debate_instruction, r, is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking3_i.content}; answer: {answer3_i.content}")
            all_thinking3[r].append(thinking3_i)
            all_answer3[r].append(answer3_i)
    final_debate = LLMAgentBase(["thinking","answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    final_instr3 = "Sub-task 3: Given all the above thinking and answers, reason over them carefully and provide a final answer."
    thinking3, answer3 = await final_debate([taskInfo, thinking1, answer1, thinking2, answer2] + all_thinking3[-1] + all_answer3[-1], final_instr3, is_sub_task=True)
    agents.append(f"Final Decision Agent, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong in your latest attempt. Using insights from previous attempts, try to solve the task better."
    cot_reflect_instruction = "Sub-task 4: Compute OC^2 for the found point C, express p/q in lowest terms, and determine p+q. " + reflect_inst
    cot_agent4 = LLMAgentBase(["thinking","answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent4 = LLMAgentBase(["feedback","correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    cot_inputs4 = [taskInfo, thinking1, answer1, thinking2, answer2, thinking3, answer3]
    subtask_desc4 = {"subtask_id":"subtask_4","instruction":cot_reflect_instruction,
                     "context":["user query",thinking1,answer1,thinking2,answer2,thinking3,answer3],"agent_collaboration":"Reflexion"}
    thinking4, answer4 = await cot_agent4(cot_inputs4, cot_reflect_instruction, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    for i in range(self.max_round):
        feedback4, correct4 = await critic_agent4([taskInfo, thinking4, answer4],
            "Please review and provide the limitations of provided solutions. If you are absolutely sure it is correct, output exactly 'True' in 'correct'", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent4.id}, feedback: {feedback4.content}; correct: {correct4.content}")
        if correct4.content == "True":
            break
        cot_inputs4.extend([thinking4, answer4, feedback4])
        thinking4, answer4 = await cot_agent4(cot_inputs4, cot_reflect_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs