async def forward_171(self, taskInfo):
    from collections import Counter
    print("Task Requirement: ", taskInfo)
    sub_tasks=[]
    agents=[]
    logs=[]
    cot_instruction="Sub-task 1: Extract the excitation ratio between star_1 and star_2 and the energy difference ΔE from the user query."
    cot_agent1=LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent",model=self.node_model,temperature=0.0)
    subtask_desc1={"subtask_id":"subtask_1","instruction":cot_instruction,"context":["user query"],"agent_collaboration":"CoT"}
    thinking1,answer1=await cot_agent1([taskInfo],cot_instruction,is_sub_task=True)
    agents.append(f"CoT agent {cot_agent1.id}, extracting values, thinking: {thinking1.content}; answer: {answer1.content}")
    sub_tasks.append(f"Sub-task 1 output: thinking - {thinking1.content}; answer - {answer1.content}")
    subtask_desc1['response']={"thinking":thinking1,"answer":answer1}
    logs.append(subtask_desc1)
    print("Step 1: ",sub_tasks[-1])
    cot_sc_instruction="Sub-task 2: Write the Boltzmann population ratio expression for each star, noting cancellation of statistical weights if equal."
    N=self.max_sc
    cot_agents2=[LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent",model=self.node_model,temperature=0.5) for _ in range(N)]
    possible_answers=[]
    thinkingmapping={}
    answermapping={}
    subtask_desc2={"subtask_id":"subtask_2","instruction":cot_sc_instruction,"context":["user query","thinking of subtask 1","answer of subtask 1"],"agent_collaboration":"SC_CoT"}
    for agent in cot_agents2:
        thinking2,answer2=await agent([taskInfo,thinking1,answer1],cot_sc_instruction,is_sub_task=True)
        agents.append(f"CoT-SC agent {agent.id}, writing Boltzmann expressions, thinking: {thinking2.content}; answer: {answer2.content}")
        possible_answers.append(answer2.content)
        thinkingmapping[answer2.content]=thinking2
        answermapping[answer2.content]=answer2
    answer2_content=Counter(possible_answers).most_common(1)[0][0]
    thinking2=thinkingmapping[answer2_content]
    answer2=answermapping[answer2_content]
    sub_tasks.append(f"Sub-task 2 output: thinking - {thinking2.content}; answer - {answer2.content}")
    subtask_desc2['response']={"thinking":thinking2,"answer":answer2}
    logs.append(subtask_desc2)
    print("Step 2: ",sub_tasks[-1])
    cot_reflect_instruction="Sub-task 3: Form the ratio of the two stars' Boltzmann factors and express the equation equal to 2."
    cot_agent3=LLMAgentBase(["thinking","answer"],"Chain-of-Thought Agent",model=self.node_model,temperature=0.0)
    critic_agent3=LLMAgentBase(["feedback","correct"],"Critic Agent",model=self.node_model,temperature=0.0)
    N_max=self.max_round
    subtask_desc3={"subtask_id":"subtask_3","instruction":cot_reflect_instruction,"context":["user query","thinking of subtask 2","answer of subtask 2"],"agent_collaboration":"Reflexion"}
    inputs3=[taskInfo,thinking2,answer2]
    thinking3,answer3=await cot_agent3(inputs3,cot_reflect_instruction,0,is_sub_task=True)
    agents.append(f"Reflexion CoT agent {cot_agent3.id}, forming ratio expression, thinking: {thinking3.content}; answer: {answer3.content}")
    for i in range(N_max):
        feedback3,correct3=await critic_agent3([taskInfo,thinking3,answer3],"Please review the ratio expression for correctness and provide feedback.",i,is_sub_task=True)
        agents.append(f"Critic agent {critic_agent3.id}, feedback: {feedback3.content}; correct: {correct3.content}")
        if correct3.content=="True":
            break
        inputs3.extend([thinking3,answer3,feedback3])
        thinking3,answer3=await cot_agent3(inputs3,cot_reflect_instruction,i+1,is_sub_task=True)
        agents.append(f"Reflexion CoT agent {cot_agent3.id}, refined ratio expression, thinking: {thinking3.content}; answer: {answer3.content}")
    sub_tasks.append(f"Sub-task 3 output: thinking - {thinking3.content}; answer - {answer3.content}")
    subtask_desc3['response']={"thinking":thinking3,"answer":answer3}
    logs.append(subtask_desc3)
    print("Step 3: ",sub_tasks[-1])
    debate_instruction4="Sub-task 4: Take the natural logarithm of the ratio expression and rearrange to isolate ln(2) in terms of T1 and T2."
    debate_agents4=[LLMAgentBase(["thinking","answer"],"Debate Agent",model=self.node_model,role=role,temperature=0.5) for role in self.debate_role]
    N_max4=self.max_round
    all_thinking4=[[] for _ in range(N_max4)]
    all_answer4=[[] for _ in range(N_max4)]
    subtask_desc4={"subtask_id":"subtask_4","instruction":debate_instruction4,"context":["user query","thinking of subtask 3","answer of subtask 3"],"agent_collaboration":"Debate"}
    for r in range(N_max4):
        for agent in debate_agents4:
            if r==0:
                thinking4,answer4=await agent([taskInfo,thinking3,answer3],debate_instruction4,r,is_sub_task=True)
            else:
                inputs4=[taskInfo,thinking3,answer3]+all_thinking4[r-1]+all_answer4[r-1]
                thinking4,answer4=await agent(inputs4,debate_instruction4,r,is_sub_task=True)
            agents.append(f"Debate agent {agent.id}, round {r}, thinking: {thinking4.content}; answer: {answer4.content}")
            all_thinking4[r].append(thinking4)
            all_answer4[r].append(answer4)
    final_decision_agent4=LLMAgentBase(["thinking","answer"],"Final Decision Agent",model=self.node_model,temperature=0.0)
    thinking4,answer4=await final_decision_agent4([taskInfo]+all_thinking4[-1]+all_answer4[-1],"Sub-task 4: Make final decision on the logarithmic rearrangement.",is_sub_task=True)
    agents.append(f"Final Decision agent {final_decision_agent4.id}, thinking: {thinking4.content}; answer: {answer4.content}")
    sub_tasks.append(f"Sub-task 4 output: thinking - {thinking4.content}; answer - {answer4.content}")
    subtask_desc4['response']={"thinking":thinking4,"answer":answer4}
    logs.append(subtask_desc4)
    print("Step 4: ",sub_tasks[-1])
    final_answer=await self.make_final_answer(thinking4,answer4,sub_tasks,agents)
    return final_answer,logs