async def forward_197(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks = []
    agents = []
    logs = []

    # Sub-task 1: SC-CoT to extract and classify data
    cot_sc_instruction = "Sub-task 1: Extract total cobalt concentration, free SCN- concentration, species involved, and cumulative stability constants β1–β4 from the user query."
    N = self.max_sc
    cot_agents = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings = []
    possible_answers = []
    subtask_desc1 = {"subtask_id": "subtask_1", "instruction": cot_sc_instruction, "context": ["user query"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        thinking1_i, answer1_i = await cot_agents[i]([taskInfo], cot_sc_instruction, i, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents[i].id}, thinking: {thinking1_i.content}; answer: {answer1_i.content}")
        possible_thinkings.append(thinking1_i)
        possible_answers.append(answer1_i)
    final_decision_agent_1 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking1, answer1 = await final_decision_agent_1([taskInfo] + possible_thinkings + possible_answers, "Sub-task 1: Synthesize and choose the most consistent extracted data.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response'] = {"thinking": thinking1, "answer": answer1}
    logs.append(subtask_desc1)
    print("Step 1: ", sub_tasks[-1])

    # Sub-task 2: Reflexion to set up equations
    reflect_inst = "Given previous attempts and feedback, carefully consider where you could go wrong and improve the setup."
    reflexion_instruction = "Sub-task 2: Set up the mass-action equations and cobalt mass balance to express each complex concentration in terms of [Co2+], β values, and [SCN-]." + reflect_inst
    cot_reflect_agent = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    critic_agent = LLMAgentBase(["feedback", "correct"], "Critic Agent", model=self.node_model, temperature=0.0)
    subtask_desc2 = {"subtask_id": "subtask_2", "instruction": reflexion_instruction, "context": ["user query", "thinking of subtask 1", "answer of subtask 1"], "agent_collaboration": "Reflexion"}
    cot_inputs2 = [taskInfo, thinking1, answer1]
    thinking2, answer2 = await cot_reflect_agent(cot_inputs2, reflexion_instruction, 0, is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_reflect_agent.id}, thinking: {thinking2.content}; answer: {answer2.content}")
    N_max = self.max_round
    for i in range(N_max):
        feedback, correct = await critic_agent([taskInfo, thinking2, answer2], "Please review and criticize the above setup. If correct, output exactly 'True' in 'correct'.", i, is_sub_task=True)
        agents.append(f"Critic agent {critic_agent.id}, feedback: {feedback.content}; correct: {correct.content}")
        if correct.content == "True":
            break
        cot_inputs2.extend([thinking2, answer2, feedback])
        thinking2, answer2 = await cot_reflect_agent(cot_inputs2, reflexion_instruction, i+1, is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_reflect_agent.id}, refined thinking: {thinking2.content}; answer: {answer2.content}")
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response'] = {"thinking": thinking2, "answer": answer2}
    logs.append(subtask_desc2)
    print("Step 2: ", sub_tasks[-1])

    # Sub-task 3: SC-CoT to perform numerical speciation
    cot_sc_instruction3 = "Sub-task 3: Solve for [Co2+], compute [Co(SCN)2], and determine the fraction α2 = [Co(SCN)2]/total Co."
    cot_agents3 = [LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.5) for _ in range(N)]
    possible_thinkings3 = []
    possible_answers3 = []
    subtask_desc3 = {"subtask_id": "subtask_3", "instruction": cot_sc_instruction3, "context": ["user query", "thinking of subtask 2", "answer of subtask 2"], "agent_collaboration": "SC_CoT"}
    for i in range(N):
        t3_i, a3_i = await cot_agents3[i]([taskInfo, thinking2, answer2], cot_sc_instruction3, i, is_sub_task=True)
        agents.append(f"CoT-SC agent {cot_agents3[i].id}, thinking: {t3_i.content}; answer: {a3_i.content}")
        possible_thinkings3.append(t3_i)
        possible_answers3.append(a3_i)
    final_decision_agent_3 = LLMAgentBase(["thinking", "answer"], "Final Decision Agent", model=self.node_model, temperature=0.0)
    thinking3, answer3 = await final_decision_agent_3([taskInfo, thinking2, answer2] + possible_thinkings3 + possible_answers3, "Sub-task 3: Synthesize and choose the most consistent speciation calculation.", is_sub_task=True)
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response'] = {"thinking": thinking3, "answer": answer3}
    logs.append(subtask_desc3)
    print("Step 3: ", sub_tasks[-1])

    # Sub-task 4: CoT to compare and select closest choice
    cot_instruction4 = "Sub-task 4: Compare the calculated fraction α2 with the provided choices (16.9%, 42.3%, 25.6%, 38.1%) and select the closest percentage."
    cot_agent4 = LLMAgentBase(["thinking", "answer"], "Chain-of-Thought Agent", model=self.node_model, temperature=0.0)
    subtask_desc4 = {"subtask_id": "subtask_4", "instruction": cot_instruction4, "context": ["user query", "thinking of subtask 3", "answer of subtask 3"], "agent_collaboration": "CoT"}
    thinking4, answer4 = await cot_agent4([taskInfo, thinking3, answer3], cot_instruction4, is_sub_task=True)
    agents.append(f"CoT agent {cot_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response'] = {"thinking": thinking4, "answer": answer4}
    logs.append(subtask_desc4)
    print("Step 4: ", sub_tasks[-1])

    final_answer = await self.make_final_answer(thinking4, answer4, sub_tasks, agents)
    return final_answer, logs